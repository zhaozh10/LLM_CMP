#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=LLM_CMP
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00

source activate win

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_luotuo.py 