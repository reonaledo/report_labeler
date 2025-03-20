# Privacy-Preserving Information Extraction Framework for Diverse Imaging Reports using Large Language Models

## Abstract
Efficient extraction of structured information from unstructured radiology reports remains a critical challenge in healthcare. We introduce the Radiology Report Information Extraction Framework (RRIEF), a privacy-preserving approach utilizing parameter-efficient fine-tuning of open-source large language models (LLMs). We validated RRIEF across chest X-ray (CXR), mammography, and coronary CT angiography (CCTA) reports, evaluating its performance against specialized methods and proprietary LLMs (GPT-4o, Gemini-1.5-Flash, Claude-3.5-Sonnet). For CXR, RRIEF-LLaMA1-65B achieved F1 scores of 0.87 and 0.85 in internal and external tests, significantly outperforming CheXpert Labeler (0.70 and 0.69, P<.001), CheXbert (0.72 and 0.69, P<.001), and all proprietary LLMs (Claude-3.5-Sonnet: 0.69 and 0.62, P<.001). For mammography, RRIEF-LLaMA1-30B/65B reached F1 scores of 0.91 and 0.99 in internal and external tests, exceeding all proprietary LLMs (0.86 and 0.92, P=.002). For CCTA, using only 100 training reports, RRIEF-LLaMA3-8B significantly outperformed Gemini-1.5-Flash in stenosis severity (0.87 vs 0.83, P=.02), GPT-4o in external testing (0.83 vs 0.68, P<.001), and all proprietary models for modifiers in external testing (1.00 vs 0.93, P=.004). Notably, RRIEF-LLaMA3-8B achieved superior performance on CXR with only 200 training samples compared to all baselines including CheXbert and proprietary LLMs (P<.001). Our locally deployable framework enables high-performance information extraction from different types of radiology reports, facilitating large-scale research and clinical practice. We provide our complete implementation code publicly to promote accessibility and adoption.

## Code Availability (CURRENTLY IN DEVELOPMENT)
The code implementation is currently in development. Both the analytical code for reproducing the results of this study and a user-friendly version that allows researchers to adapt it for their specific purposes will be made available soon in this repository.

