#!/bin/bash
#SBATCH --job-name=smoke_eval
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.err

export PYTHONUNBUFFERED=1
exec 2>&1
set -euo pipefail

echo "[$(date)] smoke_eval starting on $(hostname)"

source /etc/profile.d/modules.sh
module load cray-python/3.11.7

source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH

BASE=/work/hdd/bfmi/qlan1/cyh
cd "$BASE/software/opencompass"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

echo "[$(date)] Starting MATH-500 eval..."

python -u run.py \
    "$BASE/projects/CoT/experiments/baseline/bench_math500.py" \
    --models base_qwen25_math_7b \
    --work-dir "$BASE/outputs/eval/base"

echo "[$(date)] MATH-500 eval DONE"
