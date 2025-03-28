import os
# Force PyTorch to only see GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import logging
from typing import List, Dict, Any, Optional

import fire
import torch
import transformers
from datasets import load_dataset, Dataset, DatasetDict
import bitsandbytes as bnb
import json

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

from utils.prompter import Prompter
from utils.find_lora_target import find_all_linear_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def train(
    # model/data params
    base_model: str = "meta-llama/Llama-3.2-1B",
    data_path: str = "./data/training.json",
    data_type: str = "custom",  # custom, json, dataset
    output_dir: str = "./output",
    instruction_file: Optional[str] = None, 

    
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 0,
    train_set_size: Optional[int] = None,
    random_seed: int = 42,
    deterministic: bool = True,
    
    # lora hyperparams
    use_lora: bool = True,
    lora_r: int = 2,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Any = 'all',
    
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    
    resume_from_checkpoint: Optional[str] = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca
    load_in_4bit: bool = True,
    gradient_checkpointing: bool = False,  # Enable gradient checkpointing for memory efficiency
    cache_dir: Optional[str] = None,
):
    """
    Fine-tunes a language model with LoRA on a single GPU.
    """

    if deterministic:
        logger.info(f"Setting up deterministic training with seed {random_seed}")
        import numpy as np
        import random
        
        # Python 랜덤 시드 설정
        random.seed(random_seed)
        # Numpy 랜덤 시드 설정
        np.random.seed(random_seed)
        # PyTorch 랜덤 시드 설정
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        
        # CUDA 결정적 알고리즘 활성화
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Transformers 랜덤 시드 설정
        transformers.set_seed(random_seed)
        
        # DataLoader 워커 시드 고정
        def seed_worker(worker_id):
            worker_seed = random_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        g = torch.Generator()
        g.manual_seed(random_seed)
               
        # 환경 변수 설정
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        
        logger.info("Deterministic training setup complete")

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = batch_size // micro_batch_size
    if gradient_accumulation_steps <= 0:
        logger.warning(f"Calculated gradient_accumulation_steps={gradient_accumulation_steps} is invalid. Setting to 1.")
        gradient_accumulation_steps = 1
    
    logger.info(f"Training setup: batch_size={batch_size}, micro_batch_size={micro_batch_size}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load instrcution
    instruction_text = None
    if instruction_file:
        try:
            logger.info(f"Loading instruction from file: {instruction_file}")
            with open(instruction_file, "r", encoding="utf-8") as f:
                instruction_text = f.read().strip()
            logger.info(f"Instruction loaded successfully ({len(instruction_text)} characters)")
        except Exception as e:
            logger.error(f"Error loading instruction file: {str(e)}")
            raise

    # Log training parameters
    logger.info(
        f"Training LLM with LoRA using the following parameters:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"instruction_file: {instruction_file}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_lora: {use_lora}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"add_eos_token: {add_eos_token}\n"
        f"group_by_length: {group_by_length}\n"
        f"load_in_4bit: {load_in_4bit}\n"
        f"gradient_checkpointing: {gradient_checkpointing}\n"
    )
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if gpu_count > 1:
            if load_in_4bit:
                # For 4-bit quantization with multiple GPUs, use a special approach
                logger.warning(
                    f"Multiple GPUs detected ({gpu_count}) with 4-bit quantization. "
                    f"Due to tensor allocation requirements for 4-bit, restricting to GPU 0 only."
                )
                # Set CUDA device explicitly
                torch.cuda.set_device(0)
                # This forces all CUDA operations to go through GPU 0
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                device_map = {"": 0}
            else:
                # Use only GPU 0 by default for simplicity and consistency
                logger.info(f"Multiple GPUs detected ({gpu_count}). Using only GPU 0 for training.")
                # Force PyTorch to use only GPU 0
                torch.cuda.set_device(0)
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                device_map = {"": 0}
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            device_map = {"": 0}
    else:
        logger.warning("No GPU detected! Training on CPU will be very slow")
        device_map = "auto"

    # Load model
    try:
        logger.info(f"Loading model: {base_model}")
        
        # Configure compute dtype based on hardware
        compute_dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Use bfloat16 on Ampere or newer GPUs (sm_80 or higher)
            compute_dtype = torch.bfloat16
            logger.info("Using bfloat16 precision for computation")
        else:
            logger.info("Using float16 precision for computation")
        
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype
            )
            
            logger.info(f"Loading model with 4-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                base_model, 
                quantization_config=bnb_config, 
                device_map=device_map,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        else:
            logger.info(f"Loading model with 16-bit precision")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=compute_dtype,
                device_map=device_map,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
        # Prepare model for k-bit training if using LoRA with quantization
        if use_lora and load_in_4bit:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(model)
        
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    # Load tokenizer
    try:
        logger.info(f"Loading tokenizer from {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        tokenizer.padding_side = "left"  # Allow batched inference
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        raise

    # Configure LoRA
    if use_lora:
        if lora_target_modules == 'all':
            modules = find_all_linear_names(model, load_in_4bit)
            logger.info(f"Auto-detected LoRA target modules: {modules}")
        else:
            modules = lora_target_modules
            logger.info(f"Using specified LoRA target modules: {modules}")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        logger.info("Training whole model (LoRA disabled)")
        logger.info(f"Trainable parameters: {model.num_parameters()}")

    # Tokenization functions
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=False,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    # Initialize the prompter
    prompter = Prompter(prompt_template_name)
    
    def validate_data_point(data_point):
        """Validate that the data point has the required fields."""
        required_fields = ["input", "output"]
        for field in required_fields:
            if field not in data_point:
                raise ValueError(f"Data point is missing required field: {field}")
        return True
        
    def generate_and_tokenize_prompt(data_point):

        if instruction_text:
            data_point_instruction = instruction_text
        else:
            data_point_instruction = data_point.get("instruction", "")

        # Validate the data point structure
        try:
            validate_data_point(data_point)
        except ValueError as e:
            logger.error(f"Invalid data point: {str(e)}")
            # Return a minimal valid structure to avoid breaking the training pipeline
            return {"input_ids": [0], "attention_mask": [1], "labels": [-100]}
        
        # Generate and tokenize the prompt
        full_prompt = prompter.generate_prompt(
            data_point_instruction,
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    # Load dataset
    logger.info(f"Loading dataset from {data_path}")
    try:
        if data_type == 'custom':
            try:
                with open(data_path, "r", encoding="utf-8-sig") as f:
                    seed_tasks = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON file: {data_path}")
                raise
            except FileNotFoundError:
                logger.error(f"Dataset file not found: {data_path}")
                raise
                
            # Log dataset info
            logger.info(f"Loaded {len(seed_tasks)} examples from dataset")
            
            # Validate the first few examples to catch potential issues early
            for i, example in enumerate(seed_tasks[:5]):
                if i < min(5, len(seed_tasks)):
                    try:
                        validate_data_point(example)
                    except ValueError as e:
                        logger.warning(f"Example {i} has issue: {str(e)}")

            # Random sampling if train_set_size is specified
            if train_set_size is not None:
                import random
                random.seed(random_seed)
                if train_set_size < len(seed_tasks):
                    logger.info(f"Sampling {train_set_size} examples from dataset of size {len(seed_tasks)}")
                    seed_tasks = random.sample(seed_tasks, train_set_size)

            t = Dataset.from_list(seed_tasks)
            data = DatasetDict({"train": t})
            
        else:
            if data_path.endswith(".json") or data_path.endswith(".jsonl"):
                data = load_dataset("json", data_files=data_path)
                logger.info(f"Loaded dataset with {len(data['train'])} examples")
            else:
                data = load_dataset(data_path)
                logger.info(f"Loaded dataset from {data_path}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

    # Load checkpoint if resuming training
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = False
        
        if os.path.exists(checkpoint_name):
            logger.info(f"Restarting from checkpoint: {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            logger.error(f"Checkpoint {checkpoint_name} not found")

    # Prepare train/val splits
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=random_seed
        )
        train_data = train_val["train"].shuffle(seed=random_seed).map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle(seed=random_seed).map(generate_and_tokenize_prompt)
        logger.info(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples")
    else:
        train_data = data["train"].shuffle(seed=random_seed).map(generate_and_tokenize_prompt)
        val_data = None
        logger.info(f"Training on {len(train_data)} examples, no validation set")

    # 훈련 시작 직전 PyTorch 상태 확인 및 로깅
    if deterministic and torch.cuda.is_available():
        logger.info(f"CUDA deterministic mode: {torch.backends.cudnn.deterministic}")
        logger.info(f"CUDA benchmark mode: {torch.backends.cudnn.benchmark}")

    # # Training 인자에 generator와 worker_init_fn 추가 (DataLoader 관련 시드 고정)
    # data_loader_params = {}
    # if deterministic:
    #     data_loader_params = {
    #         "generator": g,
    #         "worker_init_fn": seed_worker
    #     }

    # Enable gradient checkpointing if requested (saves memory)
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing to save memory")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            # Fix for the deprecation warning in torch.utils.checkpoint
            if hasattr(model, "config") and hasattr(model.config, "use_reentrant"):
                model.config.use_reentrant = False
        else:
            logger.warning("Model doesn't support gradient checkpointing")

    # Initialize the trainer
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        logging_steps=4,
        optim="adamw_torch",
        eval_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=4 if val_set_size > 0 else None,
        save_steps=20,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length,
        report_to="wandb" if len(wandb_project) > 0 else None,
        run_name=wandb_run_name if len(wandb_project) > 0 else None,
        ddp_find_unused_parameters=False,
        do_eval=val_set_size > 0,
        # CRITICAL: Disable DataParallel completely for 4-bit training
        # This is the primary fix for the tensor allocation issues
        no_cuda=load_in_4bit and torch.cuda.device_count() > 1,
        # Use single device
        # With DataParallel disabled, we'll manually place on GPU 0
        dataloader_num_workers=0,  # Prevent potential issues with multiple workers
    )
    
    logger.info(f"Initializing trainer")
    
    # Additional configuration for 4-bit training with multiple GPUs
    if load_in_4bit and torch.cuda.device_count() > 1:
        logger.warning(
            "Multiple GPUs detected with 4-bit quantization. To prevent tensor allocation issues, "
            "CUDA is being disabled in the trainer and training will proceed on CPU with "
            "a single GPU handling the forward/backward passes."
        )
        # Force PyTorch to use only GPU 0
        torch.cuda.set_device(0)
        # Explicitly set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Additional settings for bitsandbytes
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # **data_loader_params
    )

    # Training preparation
    logger.info(f"Preparing for training...")
    model.config.use_cache = False
    
    # For LoRA model saving
    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
    
    # Log memory usage before training
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        logger.info(f"GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    # Save the model
    try:
        logger.info(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)
        logger.info(f"Saving tokenizer to {output_dir}")
        tokenizer.save_pretrained(output_dir)
        
        # Additional save for LoRA adapter
        if use_lora:
            adapter_path = f"{output_dir}/adapter_model.bin"
            logger.info(f"Saving LoRA adapter to {adapter_path}")
            torch.save(trainer.model.state_dict(), adapter_path)
            
        logger.info("Model and tokenizer saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

    logger.info("\nTraining completed successfully!")
    # return model
    return output_dir


if __name__ == "__main__":
    try:
        # Set PyTorch environment variables for better performance
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid deadlocks with HF tokenizers
        
        # For 4-bit training
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        
        # For reproducability
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 결정적 모드 설정
        

        # Call the training function
        fire.Fire(train)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA out of memory" in error_msg:
            logger.error(f"CUDA memory error: {error_msg}")
            logger.error(
                "Suggestions:\n"
                "1. Reduce batch_size or micro_batch_size\n"
                "2. Enable gradient_checkpointing=True\n"
                "3. Reduce model size or try a smaller model\n"
                "4. Try 4-bit quantization with load_in_4bit=True"
            )
        elif "Input tensors need to be on the same GPU" in error_msg:
            logger.error(f"GPU tensor allocation error: {error_msg}")
            logger.error(
                "This is a critical error with 4-bit quantization in multi-GPU setups.\n"
                "SOLUTION: Restart your runtime and run the script with:\n"
                "CUDA_VISIBLE_DEVICES=0 python script.py [other args...]\n"
                "This explicitly tells PyTorch to only use GPU 0."
            )
        else:
            logger.error(f"Training failed with error: {error_msg}")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise