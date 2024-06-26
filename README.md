# A Framework for Training LLMs as High-Performance, Specialized Radiology Report Labelers

## Abstract
### Background: 
Although Large Language Models (LLMs) like ChatGPT and LLaMA show promise in the medical domain, their application for radiology report labeling remains largely unexplored.
### Purpose: 
To develop and validate a general framework for a high-performance specialized report labeler.
### Materials and Methods: 
This retrospective study utilized datasets from MIMIC-CXR and Open-i for chest X-ray (CXR) reports, along with a private dataset and CDD-CESM for mammography reports. To prepare the training sets, 1,000 CXR and 500 mammography reports were annotated. The publicly accessible LLM, LLaMA, was trained for labeling across each modality, incorporating both instruction tuning and parameter-efficient fine-tuning. To investigate the impact of LLM size on the labeling performance, experiments spanned LLaMA with parameter counts ranging from 7B to 65B. Labeling performance was assessed using the Macro F1 score on both internal and external test sets (n=500/500 for CXR, n=549/326 for mammography). A comparative analysis was conducted against CheXpert, CheXbert, and zero/few-shot settings of LLaMA-65B.
### Results: 
A total of 2,000 MIMIC-CXR and Open-i reports from 1,438 patients (median age, 63 [IQR, 50-74]; 52% female) and 1,375 reports from 1,360 patients of the private datasets and CDD-CESM (median age, 51 [IQR, 46-58]; all female) were included. The proposed method achieved higher mean F1 scores than existing methods, recording 0.80 (95% CI: 0.80, 0.81) and 0.77 (95% CI: 0.76, 0.78) for CXR internal and external tests, notably outperforming CheXbert's 0.72 (95% CI: 0.71, 0.73) and 0.74 (95% CI: 0.72, 0.75), respectively (P < .001 for all comparisons). For mammography, it significantly surpassed zero/few-shot settings, achieving 0.92 (95% CI: 0.91, 0.93) and 0.97 (95% CI: 0.97, 0.98) for internal and external tests, respectively (P < .001 for all comparisons). 
### Conclusion: 
The proposed framework effectively trains an LLM to function as high-performance report labeler, tailored to the specific imaging modality and reporting style.
