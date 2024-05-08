import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
"""
import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

from utils.prompter import Prompter



def train(
    # model/data params
    modality: str = "MG",
    # modality: str = "CXR",
    # base_model: str = "huggyllama/llama-7b",  # the only required argument
    base_model: str = "llama3-8b",  # the only required argument
    # base_model: str = "meta-llama/Llama-2-7b-hf",  # the only required argument
    # base_model: str = "meta-llama/Llama-2-70b-hf",  # the only required argument
    # base_model: str = "huggyllama/llama-30b",  # the only required argument
    # base_model: str = "huggyllama/llama-65b",  # the only required argument
    model_type: str = "llama3",
    # model_type: str = "llama",
    # base_model: str = "chavinlo/alpaca-native",  # the only required argument

    # data_path: str = "yahma/alpaca-cleaned",

    data_type: str = "SNUH", # SNUH, MIMIC
    data_path: str = "/home/data/SNUH_Mammo/0518/MG_training_data_labeled_v3.2_w_inst.json",
    output_dir: str = "/home/workspace/MGLabler_trained_model/v3.2_llama3_8b_r2all_bs8",
    # data_type: str = "MIMIC", # SNUH, MIMIC
    # data_path: str = "/home/data/CXR_Labeler/CXR_training_data_v3.json",
    # output_dir: str = "/home/workspace/CXRLabler_trained_model/llama-65b_v1",

    # output_dir: str = "./no_name_folder",
    # output_dir: str = "./lora-alpaca-30b",
    # training hyperparams
    batch_size: int = 8,
    # batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # cutoff_len: int = 256,
    val_set_size: int = 32,
    # val_set_size: int = 2000,
    # lora hyperparams
    use_lora: bool = True,
    lora_r: int = 2,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules = 'all',
    # lora_target_modules = [
    #     "q_proj",
    #     "v_proj",
    # ],
    # llm hyperparams
    # train_on_inputs: bool = True,  # if False, masks out inputs in loss
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "MG_labler",
    # wandb_run_name: str = "v1_CXR_llama_65b",
    wandb_run_name: str = "no_name",
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    load_in_4bit: bool = True,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"modality: {modality}\n"
            f"base_model: {base_model}\n"
            f"model_type: {model_type}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
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
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"load_in_4bit: {load_in_4bit}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    # device_map = "auto"
    device_map = {"":0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        if model_type == 'llama':
            model = LlamaForCausalLM.from_pretrained(
                base_model, 
                quantization_config=bnb_config, 
                device_map=device_map,
                cache_dir="/home/public_models"

            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config, 
                device_map=device_map,
                cache_dir="/home/public_models"

            )
        # model.gradient_checkpointing_enable() # 연산속도 늦추고 메모리 사용량 낮춤
        if use_lora:
            model = prepare_model_for_kbit_training(model)
    else:
        if model_type == 'llama':
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
                cache_dir="/home/public_models"

            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
                cache_dir="/home/public_models"

            )

        if use_lora:
            model = prepare_model_for_int8_training(model)


    if model_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)


    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
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
            ]  # could be sped up, probably
        return tokenized_full_prompt
    
    def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit
        # cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])


        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    if use_lora:
        if lora_target_modules == 'all':
            modules = find_all_linear_names(model)
        else:
            modules = lora_target_modules

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    else:
        print("#"*20)
        print('Training whole parameters!')
        print("#"*20)

    if data_type in ['SNUH', 'MIMIC']:
        import json
        from datasets import Dataset, DatasetDict
        with open(data_path, "r", encoding="utf-8-sig") as f:
            seed_tasks = json.load(f)
        t = Dataset.from_list(seed_tasks)
        data = DatasetDict({"train": t})
    else:
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if use_lora:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    else:
        print("Trainable parameters: ",model.num_parameters())

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            # warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=use_lora,
            logging_steps=4,
            # logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=4 if val_set_size > 0 else None,
            # eval_steps=200 if val_set_size > 0 else None,
            # save_steps=4,
            save_steps=20,
            output_dir=output_dir,
            save_total_limit=100,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    torch.save(trainer.model.state_dict(), f"{output_dir}/adapter_model.bin") #  저장 안될때 

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
