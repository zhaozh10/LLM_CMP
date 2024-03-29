#!/bin/bash
#SBATCH -p bme_gpu4
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
# FILE=${FILE:-"MIMIC-EN.csv"}
TIME_SUFFIX=$(date +%Y%m%d%H%M%S)
source activate win
# source activate chatcadsource activate win

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval.py --file ${FILE} --tgt_dir ${TGT_dir} --task ${TASK}