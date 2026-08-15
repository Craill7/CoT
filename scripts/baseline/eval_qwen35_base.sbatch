#!/bin/bash
#SBATCH --job-name=eval_q35
#SBATCH --account=bgkq-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --array=0-4%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/%x_%A_%a.err

export PYTHONUNBUFFERED=1
exec 2>&1
set -euo pipefail

echo "[$(date)] eval_qwen35 task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

source /etc/profile.d/modules.sh
module load cray-python/3.11.7

source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH

BASE=/work/hdd/bfmi/qlan1/cyh
CONFIG_DIR="$BASE/projects/CoT/experiments/baseline"
cd "$BASE/software/opencompass"

BENCHMARKS=("bench_qwen35_math500.py" "bench_qwen35_aime24.py" "bench_qwen35_aime25.py" "bench_qwen35_amc23.py" "bench_qwen35_olympiad.py")
CONFIG="${CONFIG_DIR}/${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] Running ${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1

python -u run.py \
    "$CONFIG" \
    --models qwen35_9b_base \
    --work-dir "$BASE/outputs/eval/qwen35_9b_base"

echo "[$(date)] DONE"
