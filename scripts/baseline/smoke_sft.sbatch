#!/bin/bash
#SBATCH --job-name=smoke_sft
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.err

export PYTHONUNBUFFERED=1
exec 2>&1
set -euo pipefail

echo "[$(date)] smoke_sft starting on $(hostname)"

source /etc/profile.d/modules.sh
module load cray-python/3.11.7

source /work/hdd/bfmi/qlan1/cyh/software/envs/llama/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE=/work/hdd/bfmi/qlan1/cyh
cd "$BASE/software/LLaMA-Factory"

export LLAMAFACTORY_DATASET_INFO="$BASE/projects/CoT/experiments/baseline/dataset_info_test.json"

echo "[$(date)] Starting SFT 50-sample training..."

torchrun --nproc_per_node=2 src/train.py \
    "$BASE/projects/CoT/experiments/baseline/train_raw_sft_test.yaml"

echo "[$(date)] SFT smoke DONE"
