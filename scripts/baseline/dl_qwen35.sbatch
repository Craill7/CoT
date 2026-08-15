#!/bin/bash
#SBATCH --job-name=dl_qwen35
#SBATCH --account=bgqj-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/%x_%j.err

export PYTHONUNBUFFERED=1
exec 2>&1
set -euo pipefail

echo "[$(date)] Starting Qwen3.5-9B-Base download on $(hostname)"

source /etc/profile.d/modules.sh
module load cray-python/3.11.7

source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export HF_HOME=/work/hdd/bfmi/qlan1/cyh/tmp/hf_cache
mkdir -p "$HF_HOME"

MODEL_DIR=/work/hdd/bfmi/qlan1/cyh/data/models/Qwen3.5-9B-Base
mkdir -p "$MODEL_DIR"

echo "[$(date)] Downloading from HuggingFace..."
hf download Qwen/Qwen3.5-9B-Base --local-dir "$MODEL_DIR"

echo "[$(date)] Download complete. Checking files..."
ls -lh "$MODEL_DIR"
du -sh "$MODEL_DIR"

echo "[$(date)] DONE"
