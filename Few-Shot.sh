#!/bin/bash
#SBATCH -p bme_gpu_fat
#SBATCH --job-name=LLM_CMP_${TIME_SUFFIX}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00

set -x
FILE=$1
TGT_dir=$2
TASK=$3
SHOT=$4
TIME_SUFFIX=$(date +%Y%m%d%H%M%S)
source activate win

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python few_shot_eval.py --file ${FILE} --tgt_dir ${TGT_dir} --task ${TASK} --shot ${SHOT}