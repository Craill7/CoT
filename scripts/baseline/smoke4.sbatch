#!/bin/bash
#SBATCH --job-name=smoke4
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=32G --time=00:30:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/smoke4_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/smoke4_%j.err

export PYTHONUNBUFFERED=1
set -euo pipefail

source /etc/profile.d/modules.sh
module load cray-python/3.11.7
source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export HF_HUB_OFFLINE=1

echo "[$(date)] Test 1: import datasets (huggingface)"
timeout 30 python3 -u -c "import datasets; print('OK:', datasets.__version__)" 2>&1 || echo "TIMEOUT"

echo "[$(date)] Test 2: import huggingface_hub"
timeout 30 python3 -u -c "import huggingface_hub; print('OK:', huggingface_hub.__version__)" 2>&1 || echo "TIMEOUT"

echo "[$(date)] Test 3: with script (fake TTY)"
timeout 60 script -q -c "python3 -u -c 'from opencompass.datasets import CustomDataset; print(\"OK\")'" /dev/null 2>&1 || echo "TIMEOUT"

echo "[$(date)] Test 4: with script (VLLM)"
timeout 60 script -q -c "python3 -u -c 'from opencompass.models import VLLM; print(\"OK\")'" /dev/null 2>&1 || echo "TIMEOUT"

echo "[$(date)] DONE"
