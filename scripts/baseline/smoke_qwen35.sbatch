#!/bin/bash
#SBATCH --job-name=smoke_q35
#SBATCH --account=bgkq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.err

export PYTHONUNBUFFERED=1
exec 2>&1
set -euo pipefail

echo "[$(date)] smoke qwen35 on $(hostname)"

source /etc/profile.d/modules.sh
module load cray-python/3.11.7

source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/work/hdd/bfmi/qlan1/cyh/tmp/cache

cd /work/hdd/bfmi/qlan1/cyh/software/opencompass

# Use HuggingFaceCausalLM instead of vLLM to avoid Qwen3.5-VL misidentification
python -u run.py \
    /work/hdd/bfmi/qlan1/cyh/projects/CoT/experiments/baseline/bench_qwen35_aime24_hf.py \
    --models qwen35_9b_base_hf \
    --work-dir /work/hdd/bfmi/qlan1/cyh/outputs/eval/qwen35_9b_base

echo "[$(date)] DONE"
