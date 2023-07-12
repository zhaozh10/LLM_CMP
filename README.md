# LLM_CMP

-  Switch to your LLM: Change corresponding code in eval.py to prepare your tokenizer, model and generation_config
- To implement it on a cluster, you need to implement LLM.sh and modify it based on your condition
- All data has been well prepared in this repo, just clone this repo
- Now only support ImpressionGPT, the results from LLM will be recorded in the ''pseudo_impression'' column of the csv file

Implement Ziya-13B on ImpressionGPT 
- sbatch LLM.sh MIMIC-EN.csv Ziya-13B ImpressionGPT
- sbatch LLM.sh MIMIC-ZH.csv Ziya-13B ImpressionGPT
- sbatch LLM.sh Open_I-EN.csv Ziya-13B ImpressionGPT
- sbatch LLM.sh Open_I-ZH.csv Ziya-13B ImpressionGPT

Implement Ziya-13B on RadQNLI
- sbatch LLM.sh RadQNLI-EN.csv Ziya-13B RadQNLI
- sbatch LLM.sh RadQNLI-ZH.csv Ziya-13B RadQNLI

Implement Ziya-13B on DeID
- sbatch LLM.sh DeID.csv Ziya-13B DeID


sbatch Few-Shot.sh MIMIC-EN.csv Ziya-13B ImpressionGPT one-shot