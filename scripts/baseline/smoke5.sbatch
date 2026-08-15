#!/bin/bash
#SBATCH --job-name=smoke5
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=32G --time=00:30:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/smoke5_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/smoke5_%j.err

export PYTHONUNBUFFERED=1
set -euo pipefail

source /etc/profile.d/modules.sh
module load cray-python/3.11.7
source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

echo "[$(date)] Trace opencompass.models.VLLM import (last 200 lines of -v)"
timeout 30 python3 -v -c "from opencompass.models import VLLM" 2>&1 | tail -200 || echo "EXIT: $?"
echo "[$(date)] DONE"
