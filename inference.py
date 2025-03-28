import argparse
import json
import os
import random
from typing import List

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from modality_tools import *
from alpaca_lora.utils.prompter import Prompter
from alpaca_lora.utils.find_lora_target import find_all_linear_names


def divide_batch(data_list: list, batch_size: int):
    """Split list into batches."""
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]


def load_model_and_tokenizer(model_id: str, device_id: int, cache_dir: str = None):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    
    # Set padding token and side
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": device_id},
        cache_dir=cache_dir
    )
    
    model.eval()
    model.config.use_cache = True  # Enable cache for inference
    
    return model, tokenizer


def load_peft_model(base_model, checkpoint_path: str, lora_r: int, device_id: int):
    """Load PEFT model with LoRA adapters."""
    lora_target_modules = find_all_linear_names(base_model, load_in_4bit=True)
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=16,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(base_model, config)
    set_peft_model_state_dict(model, torch.load(checkpoint_path, map_location=torch.device(f"cuda:{device_id}")))
    
    return model


def generate_labels(model, tokenizer, prompts: List[str], batch_size: int, 
                    report_type: str, max_new_tokens: int = 100):
    """Generate labels for medical reports."""
    batches = list(divide_batch(prompts, batch_size))
    all_labels = []
    
    for batch in tqdm(batches):
        with torch.no_grad():
            generated_ids = model.generate(
                **tokenizer(
                    batch,
                    return_tensors='pt',
                    return_token_type_ids=False,
                    truncation=False,
                    padding=True,
                ).to(f"cuda:{args.device}"),
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=['}'],
                tokenizer=tokenizer
            )
            
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Post-process based on report type
        if report_type == 'ccta':
            labels = post_process_ccta_labels(generated_texts)
        elif report_type == 'cxr':
            labels = post_process_cxr_labels(generated_texts, is_zeroshot=False)
        elif report_type == 'mg':
            labels = post_process_mg_labels(generated_texts)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
            
        all_labels.extend(labels)
        
    return all_labels


def main(args):
    """Main function to run the inference pipeline."""
    # Set random seed
    random.seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device, args.cache_dir)
    
    # Load PEFT model
    model = load_peft_model(model, args.checkpoint, args.lora_r, args.device)
    
    # Load input data
    data_ext = os.path.splitext(args.input_file)[1].lower()
    if data_ext == '.csv':
        data = pd.read_csv(args.input_file)
    elif data_ext in ['.xlsx', '.xls']:
        data = pd.read_excel(args.input_file)
    elif data_ext == '.json':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError(f"Unsupported file format: {data_ext}")
    
    # Check if the specified input column exists in the data
    if args.input_col not in data.columns:
        raise ValueError(f"Input column '{args.input_col}' not found in data. Available columns: {list(data.columns)}")
    
    # Load instruction prompt
    with open(args.instruction_file, 'r', encoding='utf-8') as f:
        instruction = f.read()

    # Generate prompts
    prompter = Prompter('alpaca')
    prompts = [prompter.generate_prompt(instruction, f) for f in data[args.input_col]]
    
    # Generate labels
    max_tokens = 100
    if args.report_type == 'cxr':
        max_tokens = 80
    elif args.report_type == 'mg':
        max_tokens = 140
    
    labels = generate_labels(model, tokenizer, prompts, args.batch_size, 
                             args.report_type, max_tokens)
    
    # Process results based on report type
    if args.report_type == 'cxr':
        # Process CXR results to add them directly to the dataframe
        result_df = process_cxr_results(labels, data)
    elif args.report_type == 'mg':
        # Process MG results
        if args.standardize_keys:
            labels = standardize_mg_keys(labels)
        result_df = pd.concat([data, pd.DataFrame(labels)], axis=1)
    else:
        # CCTA results are already in the right format
        result_df = pd.concat([data, pd.DataFrame(labels)], axis=1)
    
    # Save results to a single output file
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    
    output_ext = os.path.splitext(args.output_file)[1].lower()
    if output_ext == '.csv':
        result_df.to_csv(args.output_file, index=False)
    elif output_ext in ['.xlsx', '.xls']:
        result_df.to_excel(args.output_file, index=False)
    else:
        result_df.to_csv(args.output_file, index=False)
    
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Report Analysis using LLMs")
    
    # Model parameters
    parser.add_argument("--model_id", type=str, required=True, 
                        help="Model ID (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache downloaded models")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID to use")
    
    # Fine-tuning parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the checkpoint to load")
    parser.add_argument("--lora_r", type=int, default=2,
                        help="LoRA rank parameter")
    
    # Input/output parameters
    parser.add_argument("--report_type", type=str, required=True, choices=["ccta", "cxr", "mg"],
                        help="Type of medical report to analyze")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input data file")
    parser.add_argument("--input_col", type=str, required=True,
                        help="Column name in the input file containing the text to analyze")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save output results")
    parser.add_argument("--instruction_file", type=str, default=None,
                        help="Path to instruction prompt file")
    
    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=1125,
                        help="Random seed for reproducibility")
    parser.add_argument("--standardize_keys", action="store_true",
                        help="Whether to standardize keys in MG results")
    
    args = parser.parse_args()
    
    main(args)