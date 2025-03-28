from typing import List
import pandas as pd


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


def process_cxr_results(results, data):
    """Process CXR inference results and add them as columns to the original data."""
    import ast
    
    # Define column name mapping
    column_mapping = {
        'EC': 'Enlarged Cardiomediastinum',
        'Ca': 'Cardiomegaly',
        'LL': 'Lung Lesion',
        'LO': 'Lung Opacity',
        'E': 'Edema',
        'Co': 'Consolidation',
        'P': 'Pneumonia',
        'A': 'Atelectasis',
        'Px': 'Pneumothorax',
        'PE': 'Pleural Effusion',
        'PO': 'Pleural Other',
        'F': 'Fracture',
        'SD': 'Support Devices'
    }
    
    # Process each result string into a standardized dictionary
    processed_results = []
    for i, item in enumerate(results):
        try:
            # Convert string to dictionary
            dic = ast.literal_eval(item)
            if isinstance(dic, dict):
                # Create a new dictionary with mapped column names
                mapped_dic = {}
                
                # First standardize values and map column names
                for key in dic:
                    # Get the mapped column name or use the original if not in mapping
                    mapped_key = column_mapping.get(key, key)
                    
                    # Standardize values
                    if dic[key] == 'No':
                        mapped_dic[mapped_key] = '0'
                    elif dic[key] == 'Yes':
                        mapped_dic[mapped_key] = '1'
                    elif dic[key] == 'Maybe':
                        mapped_dic[mapped_key] = '-1'
                    elif dic[key] == 'Undefined':
                        mapped_dic[mapped_key] = '2'
                    else:
                        mapped_dic[mapped_key] = dic[key]
                
                processed_results.append(mapped_dic)
            else:
                # Add empty result for invalid entries
                processed_results.append({})
        except:
            # Add empty result for parsing errors
            processed_results.append({})
    
    # Create a DataFrame with the processed results
    result_df = pd.DataFrame(processed_results)
    
    # Return the combined DataFrame (original data + processed results)
    return pd.concat([data, result_df], axis=1)


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


def post_process_cxr_labels(texts: List[str], is_zeroshot: bool):
    """Post-process CXR (Chest X-Ray) labels."""
    if is_zeroshot:
        texts = [text.split('Answer according to the template:')[-1].split("}")[0] + '}' for text in texts]
        texts = [text.replace('\n', '') for text in texts]
    else:
        texts = [text.split('### Response:')[-1].strip() for text in texts]
    
    return texts


def post_process_mg_labels(texts: List[str]):
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


def post_process_ccta_labels(texts: List[str]):
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
