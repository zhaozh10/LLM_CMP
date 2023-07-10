#!/bin/bash
#SBATCH -p bme_gpu_fat
#SBATCH --job-name=LLM_CMP_${TIME_SUFFIX}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python apply_delta.py --base /public_bme/data/llm/llama-13b --target /public_bme/data/llm/Ziya-LLaMA-13B --delta /public_bme/data/llm/Ziya-LLaMA-13B-v1
