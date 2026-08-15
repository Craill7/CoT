#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --account=bgqj-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bfmi/qlan1/cyh/logs/diag_%j.out
#SBATCH --error=/work/hdd/bfmi/qlan1/cyh/logs/diag_%j.err

# Force unbuffered
export PYTHONUNBUFFERED=1
exec 2>&1

echo "[$(date)] START diag on $(hostname)"

echo "[$(date)] Step 1: source modules.sh"
source /etc/profile.d/modules.sh
echo "[$(date)] Step 1: OK"

echo "[$(date)] Step 2: module load cray-python"
module load cray-python/3.11.7
echo "[$(date)] Step 2: OK, python=$(which python3)"

echo "[$(date)] Step 3: activate eval_cu128"
source /work/hdd/bfmi/qlan1/cyh/software/envs/eval_cu128/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
echo "[$(date)] Step 3: OK, python=$(which python3)"

echo "[$(date)] Step 4: import torch"
python3 -u -c "import torch; print(f'torch {torch.__version__}, cuda {torch.version.cuda}, available: {torch.cuda.is_available()}')"
echo "[$(date)] Step 4: OK"

echo "[$(date)] Step 5: import vllm"
python3 -u -c "import vllm; print(f'vllm {vllm.__version__}')"
echo "[$(date)] Step 5: OK"

echo "[$(date)] Step 6: OpenCompass import"
cd /work/hdd/bfmi/qlan1/cyh/software/opencompass
python3 -u -c "from opencompass.models import VLLM; print('OpenCompass VLLM import OK')"
echo "[$(date)] Step 6: OK"

echo "[$(date)] ALL STEPS PASSED"
