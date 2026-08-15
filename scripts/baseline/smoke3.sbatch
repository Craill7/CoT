#!/bin/bash
#SBATCH --job-name=smoke3
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=32G --time=00:30:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/smoke3_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/smoke3_%j.err

export PYTHONUNBUFFERED=1
set -euo pipefail

source /etc/profile.d/modules.sh
module load cray-python/3.11.7
source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# HF offline mode — prevent network hangs
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[$(date)] Test 1: CustomDataset with HF offline"
timeout 60 python3 -u -c "from opencompass.datasets import CustomDataset; print('OK')" 2>&1 || echo "TIMEOUT"

echo "[$(date)] Test 2: VLLM model with HF offline"
timeout 60 python3 -u -c "from opencompass.models import VLLM; print('OK')" 2>&1 || echo "TIMEOUT"

echo "[$(date)] DONE — imports passed, ready for full eval"
