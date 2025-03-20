import argparse
import json
import os
import random
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def find_all_linear_names(model):
    """Find all linear layer names for LoRA targeting."""
    import bitsandbytes as bnb
    
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


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
    lora_target_modules = find_all_linear_names(base_model)
    
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
                    report_type: str, is_zeroshot: bool = False, max_new_tokens: int = 100):
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
            labels = post_process_ccta_labels(generated_texts, is_zeroshot)
        elif report_type == 'cxr':
            labels = post_process_cxr_labels(generated_texts, is_zeroshot)
        elif report_type == 'mg':
            labels = post_process_mg_labels(generated_texts, is_zeroshot)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
            
        all_labels.extend(labels)
        
    return all_labels


def post_process_ccta_labels(texts: List[str], is_zeroshot: bool):
    """Post-process CCTA (Coronary CT Angiography) labels."""
    default_dict = {
        "CAD-RADS": "z",
        "Plaque_Burden": "z",
        "S": "z",
        "HRP": "z",
        "G": "z",
        "N": "z",
        "I": "z",
        "E": "z"
    }
    
    processed_labels = []
    for text in texts:
        text = text.split('### Response:')[-1].strip()
        
        try:
            result = eval(text)
            if isinstance(result, dict):
                processed_labels.append(result)
            else:
                processed_labels.append(default_dict)
        except:
            processed_labels.append(default_dict)
            
    return processed_labels


def post_process_cxr_labels(texts: List[str], is_zeroshot: bool):
    """Post-process CXR (Chest X-Ray) labels."""
    if is_zeroshot:
        texts = [text.split('Answer according to the template:')[-1].split("}")[0] + '}' for text in texts]
        texts = [text.replace('\n', '') for text in texts]
    else:
        texts = [text.split('### Response:')[-1].strip() for text in texts]
    
    return texts


def post_process_mg_labels(texts: List[str], is_zeroshot: bool):
    """Post-process MG (Mammography) labels."""
    default_dict = {
        'N': 'z',
        'M': 'z',
        'C': 'z',
        'A': 'z',
        'AD': 'z',
        'ST': 'z',
        'LNE': 'z',
        'ILN': 'z',
        'NR': 'z',
        'SR': 'z',
        'TT': 'z'
    }
    
    processed_labels = []
    for text in texts:
        text = text.split('### Response:')[-1].strip()
        
        # Replace common text
        text = text.replace('right', 'r')
        text = text.replace('left', 'l')
        text = text.replace('both', 'b')
        text = text.replace('obscure', 'o')
        
        try:
            result = eval(text)
            if isinstance(result, dict):
                processed_labels.append(result)
            else:
                processed_labels.append(default_dict)
        except:
            processed_labels.append(default_dict)
            
    return processed_labels


def standardize_mg_keys(data_list):
    """Standardize mammography data keys."""
    key_mapping = {
        'N': 'Nodule',
        'M': 'Mass',
        'C': 'Calcification',
        'A': 'Asymmetry',
        'AD': 'Architectural Distortion',
        'ST': 'Skin Thickening',
        'LNE': 'Lymph Node Enlargement',
        'ILN': 'Intramammary Lymph Node',
        'NR': 'Nipple Retraction',
        'SR': 'Skin Retraction',
        'TT': 'Trabecular Thickening'
    }
    
    standardized_list = []
    for item in data_list:
        standardized_item = {}
        for key, value in item.items():
            if key in key_mapping:
                standardized_item[key_mapping[key]] = value
            else:
                standardized_item[key] = value
        standardized_list.append(standardized_item)
    
    return standardized_list


def create_prompts(data, instruction, report_type, use_finetuned, n_shot=0, train_data=None):
    """Create prompts based on report type and training approach."""
    if use_finetuned:
        # Fine-tuned prompt creation
        if report_type == 'ccta':
            return [f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{f}\n\n### Response:\n" for f in data['input']]
        elif report_type == 'cxr':
            return [f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input: {f}\n\n### Response:" for f in data['Findings']]
        elif report_type == 'mg':
            return [f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input: {f}\n\n### Response:" for f in data['processed_finding']]
    
    elif n_shot > 0 and train_data:
        # Few-shot prompt creation
        prompts = []
        bars = '-' * 10
        
        if report_type == 'ccta':
            input_col = 'input'
            for f in data['input']:
                eg = random.sample(train_data, k=n_shot)
                examples = ""
                for k in range(n_shot):
                    text = f"""Example{k+1}:\n### Input:\n{eg[k]['input']}\n\n### Response: {eg[k]['output']}\n"""
                    examples += text + "\n"
                
                pt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n{bars}\n{examples}{bars}\n\n### Input:\n{f}\n\n### Response:\n{{'CAD-RADS': '"""
                prompts.append(pt)
        
        elif report_type == 'cxr':
            # Implement CXR-specific few-shot prompt creation
            input_col = 'Findings'
            for f in data[input_col]:
                eg = random.sample(train_data, k=n_shot)
                examples = ""
                for k in range(n_shot):
                    eg_output = transform_cxr_output(eg[k]['output'])
                    text = f"""Example{k+1}:\nReport: {eg[k]['input']}\nAnswer according to the template: {eg_output}"""
                    examples += text + "\n"
                
                pt = f"""As an AI trained in analyzing radiology reports, your task is to classify the presence or absence of specific findings in each report. Respond according to the given template. 
Default to 'Undefined' if the finding is not explicitly mentioned. If a finding is explicitly mentioned as present, mark 'Yes'. If a finding is explicitly mentioned as absent, mark 'No'. If a finding is explicitly mentioned but unclear, mark 'Maybe'. 

Template:
{{"Atelectasis": "[ANSWER]", "Cardiomegaly": "[ANSWER]", "Consolidation": "[ANSWER]", "Edema": "[ANSWER]", "Enlarged Cardiomediastinum": "[ANSWER]", "Fracture": "[ANSWER]", "Lung Lesion": "[ANSWER]", "Lung Opacity": "[ANSWER]", "Pleural Effusion": "[ANSWER]", "Pleural Other": "[ANSWER]", "Pneumonia": "[ANSWER]", "Pneumothorax": "[ANSWER]", "Support Devices": "[ANSWER]"}}

{examples}

Report: {f}
Answer according to the template: {{"Atelectasis": """
                prompts.append(pt)
        
        elif report_type == 'mg':
            input_col = 'processed_finding'
            for f in data[input_col]:
                eg = random.sample(train_data, k=n_shot)
                examples = ""
                for k in range(n_shot):
                    eg_output = reformat_mg_output(eg[k]['output'])
                    text = f"""Example{k+1}: ### Input: {eg[k]['input']}\n### Response: {eg_output}"""
                    examples += text + "\n"

                pt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n{bars}\n{examples}{bars}\n\n### Input: {f}\n### Response:"""
                prompts.append(pt)
        
        return prompts
    
    else:
        # Zero-shot prompt creation
        if report_type == 'ccta':
            return [f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input: {f}\n### Response:""" for f in data['processed_finding']]
        
        elif report_type == 'cxr':
            zero_shot_template = """As an AI trained in analyzing radiology reports, your task is to classify the presence or absence of specific findings in each report. Respond according to the given template. 
Default to 'Undefined' if the finding is not explicily mentioned. If a finding is explicily mentioned as present, mark 'Yes'. If a finding is explicily mentioned as absent, mark 'No'. If a finding is explicily mentioned but unclear, mark 'Maybe'. 

Template:
{
'Atelectasis': '[ANSWER]',
'Cardiomegaly': '[ANSWER]',
'Consolidation': '[ANSWER]',
'Edema': '[ANSWER]',
'Enlarged Cardiomediastinum': '[ANSWER]',
'Fracture': '[ANSWER]',
'Lung Lesion': '[ANSWER]',
'Lung Opacity': '[ANSWER]',
'Pleural Effusion': '[ANSWER]',
'Pleural Other': '[ANSWER]',
'Pneumonia': '[ANSWER]',
'Pneumothorax': '[ANSWER]',
'Support Devices': '[ANSWER]',
}

Report:
"""
            return [zero_shot_template + f + """

Answer according to the template:
{
'Atelectasis': '""" for f in data['Findings']]
        
        elif report_type == 'mg':
            return [f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input: {f}\n### Response:""" for f in data['processed_finding']]


def transform_cxr_output(output_str):
    """Transform CXR output format for few-shot learning."""
    import ast
    
    # Key mapping for CXR
    key_mapping = {
        'A': 'Atelectasis',
        'Ca': 'Cardiomegaly',
        'LL': 'Lung Lesion',
        'LO': 'Lung Opacity',
        'E': 'Edema',
        'Co': 'Consolidation',
        'P': 'Pneumonia',
        'Px': 'Pneumothorax',
        'PE': 'Pleural Effusion',
        'PO': 'Pleural Other',
        'F': 'Fracture',
        'SD': 'Support Devices',
        'EC': 'Enlarged Cardiomediastinum'
    }
    
    # Convert string to dictionary
    output_dict = ast.literal_eval(output_str)
    new_output = {}
    for key in key_mapping:
        if key in output_dict:
            new_output[key_mapping[key]] = output_dict[key]
    
    # Convert back to string with replacements
    dict_output = json.dumps(new_output)
    dict_output = dict_output.replace('-1', 'Maybe')
    dict_output = dict_output.replace('1', 'Yes')
    dict_output = dict_output.replace('0', 'No')
    dict_output = dict_output.replace('-2', 'Undefined')
    
    return dict_output


def reformat_mg_output(text):
    """Reformat mammography output for few-shot learning."""
    text = text.replace("'N'", "'Nodule'")
    text = text.replace("'M'", "'Mass'")
    text = text.replace("'C'", "'Calcification'")
    text = text.replace("'A'", "'Asymmetry'")
    text = text.replace("'AD'", "'Architectural Distortion'")
    text = text.replace("'ST'", "'Skin Thickening'")
    text = text.replace("'LNE'", "'Lymph Node Enlargement'")
    text = text.replace("'ILN'", "'Intramammary Lymph Node'")
    text = text.replace("'NR'", "'Nipple Retraction'")
    text = text.replace("'SR'", "'Skin Retraction'")
    text = text.replace("'TT'", "'Trabecular Thickening'")
    return text


def process_cxr_results(results, output_path):
    """Process and save CXR inference results."""
    import ast
    
    # Convert string results to dictionaries
    dicts = []
    error_indices = []
    
    for i, item in enumerate(results):
        try:
            # Convert string to dictionary
            dic = ast.literal_eval(item)
            if isinstance(dic, dict):
                dicts.append(dic)
            else:
                error_indices.append(i)
        except:
            error_indices.append(i)
    
    # Create DataFrame from valid dictionaries
    df = pd.DataFrame(dicts)
    
    # Standardize values
    df = df.replace('-2', '2')
    df = df.replace('No', '0')
    df = df.replace('Yes', '1')
    df = df.replace('Maybe', '-1')
    df = df.replace('Undefined', '2')
    
    # Save processed results
    processed_path = os.path.join(os.path.dirname(output_path), f"processed_{os.path.basename(output_path)}")
    df.to_csv(processed_path, index=False)
    print(f"Processed results saved to: {processed_path}")
    print(f"Error indices: {error_indices}")
    
    return df


def main(args):
    """Main function to run the inference pipeline."""
    # Set random seed
    random.seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device, args.cache_dir)
    
    # Load PEFT model if using fine-tuned model
    if args.use_finetuned:
        model = load_peft_model(model, args.checkpoint, args.lora_r, args.device)
    
    # Load instruction
    if args.instruction_file:
        with open(args.instruction_file, 'r', encoding='utf-8') as f:
            instruction = f.read()
    else:
        # Use default instruction based on report type
        if args.report_type == 'ccta':
            instruction = """You are an assistant designed to extract coronary artery disease information from cardiac CT reports.
            Users will send a report and you have to indicate the presence or absence of the following observations:
            
            1. CAD-RADS Score (0-5, N for non-diagnostic)
            2. Plaque Burden (P0-P4, None if not mentioned)
            3. S (Stent present: 1 if present, 0 if not)
            4. HRP (High-risk plaque: 1 if present, 0 if not)
            5. G (Graft: 1 if present, 0 if not)
            6. N (Non-diagnostic segments: 1 if present, 0 if not)
            7. I (Interarterial course: 1 if present, 0 if not)
            8. E (Other clinically important extra-coronary findings: 1 if present, 0 if not)
            """
        elif args.report_type == 'cxr':
            instruction = """You are an assistant designed to analyze chest X-ray reports.
            Users will send a report and you have to indicate the presence or absence of the following findings:
            
            - Atelectasis
            - Cardiomegaly
            - Consolidation
            - Edema
            - Enlarged Cardiomediastinum
            - Fracture
            - Lung Lesion
            - Lung Opacity
            - Pleural Effusion
            - Pleural Other
            - Pneumonia
            - Pneumothorax
            - Support Devices
            """
        elif args.report_type == 'mg':
            instruction = """You are an assistant designed to analyze mammography reports.
            Users will send a report and you have to indicate the presence or absence of the following findings:
            
            - N (Nodule)
            - M (Mass)
            - C (Calcification)
            - A (Asymmetry)
            - AD (Architectural Distortion)
            - ST (Skin Thickening)
            - LNE (Lymph Node Enlargement)
            - ILN (Intramammary Lymph Node)
            - NR (Nipple Retraction)
            - SR (Skin Retraction)
            - TT (Trabecular Thickening)
            """
    
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
    
    # Load training data for few-shot prompts
    train_data = None
    if not args.use_finetuned and args.n_shot > 0:
        if args.train_file:
            with open(args.train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
        else:
            raise ValueError("Training data is required for few-shot prompting")
    
    # Create prompts
    is_zeroshot = not args.use_finetuned and args.n_shot == 0
    prompts = create_prompts(data, instruction, args.report_type, 
                             args.use_finetuned, args.n_shot, train_data)
    
    # Generate labels
    if args.report_type == 'ccta':
        max_tokens = 100
    elif args.report_type == 'cxr':
        max_tokens = 150 if is_zeroshot else 80
    elif args.report_type == 'mg':
        max_tokens = 140
    
    labels = generate_labels(model, tokenizer, prompts, args.batch_size, 
                             args.report_type, is_zeroshot, max_tokens)
    
    # Post-process labels if needed
    if args.report_type == 'mg' and args.standardize_keys:
        labels = standardize_mg_keys(labels)
    
    # Save results
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    
    result_df = pd.concat([data, pd.DataFrame(labels)], axis=1)
    
    output_ext = os.path.splitext(args.output_file)[1].lower()
    if output_ext == '.csv':
        result_df.to_csv(args.output_file, index=False)
    elif output_ext in ['.xlsx', '.xls']:
        result_df.to_excel(args.output_file, index=False)
    else:
        result_df.to_csv(args.output_file, index=False)
    
    print(f"Results saved to: {args.output_file}")
    
    # Additional processing for CXR results
    if args.report_type == 'cxr' and not is_zeroshot:
        process_cxr_results(labels, args.output_file)


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
    parser.add_argument("--use_finetuned", action="store_true",
                        help="Whether to use a fine-tuned model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the checkpoint to load")
    parser.add_argument("--lora_r", type=int, default=2,
                        help="LoRA rank parameter")
    
    # Few-shot parameters
    parser.add_argument("--n_shot", type=int, default=0,
                        help="Number of examples for few-shot prompting")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Path to training data for few-shot examples")
    
    # Input/output parameters
    parser.add_argument("--report_type", type=str, required=True, choices=["ccta", "cxr", "mg"],
                        help="Type of medical report to analyze")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input data file")
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
    
    # Verify arguments
    if args.use_finetuned and args.checkpoint is None:
        parser.error("--checkpoint is required when --use_finetuned is set")
    
    if not args.use_finetuned and args.n_shot > 0 and args.train_file is None:
        parser.error("--train_file is required for few-shot learning")
    
    main(args)