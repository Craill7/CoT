#!/bin/bash
#SBATCH --job-name=smoke2
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 --mem=32G --time=00:30:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/smoke2_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/smoke2_%j.err

export PYTHONUNBUFFERED=1
set -euo pipefail

source /etc/profile.d/modules.sh
module load cray-python/3.11.7
source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "[$(date)] Test 1: import opencompass (timeout 120s)"
timeout 120 python3 -u -c "import opencompass; print('opencompass OK:', opencompass.__version__)" 2>&1 || echo "TIMEOUT/FAILED"

echo "[$(date)] Test 2: import opencompass.datasets (timeout 120s)"
timeout 120 python3 -u -c "from opencompass.datasets import CustomDataset; print('CustomDataset OK')" 2>&1 || echo "TIMEOUT/FAILED"

echo "[$(date)] Test 3: import opencompass.models.VLLM (timeout 120s)"
timeout 120 python3 -u -c "from opencompass.models import VLLM; print('VLLM OK')" 2>&1 || echo "TIMEOUT/FAILED"

echo "[$(date)] Test 4: import opencompass.cli.run (timeout 120s)"
timeout 120 python3 -u -c "from opencompass.cli import run; print('run module OK')" 2>&1 || echo "TIMEOUT/FAILED"

echo "[$(date)] ALL TESTS DONE"
