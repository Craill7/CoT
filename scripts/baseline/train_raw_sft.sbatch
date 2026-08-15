#!/bin/bash
#SBATCH --job-name=sft_raw
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.err

export PYTHONUNBUFFERED=1
exec 2>&1
set -euo pipefail

echo "[$(date)] sft_raw starting on $(hostname)"

source /etc/profile.d/modules.sh
module load cray-python/3.11.7

source /work/hdd/bfmi/qlan1/cyh/software/envs/llama/bin/activate
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

BASE=/work/hdd/bfmi/qlan1/cyh
cd "$BASE/software/LLaMA-Factory"

export LLAMAFACTORY_DATASET_INFO="$BASE/projects/CoT/experiments/baseline/dataset_info.json"

echo "[$(date)] Starting Raw CoT SFT training (3k samples, 2 GPU, ZeRO-2+offload)..."

torchrun --nproc_per_node=2 src/train.py \
    "$BASE/projects/CoT/experiments/baseline/train_raw_sft.yaml"

echo "[$(date)] sft_raw DONE"
