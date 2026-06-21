#!/bin/bash
#SBATCH --job-name=porecdft_vext_gpu
#SBATCH --partition=sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/vext_gpu_%j.out
#SBATCH --error=logs/vext_gpu_%j.err

# Build the rotation-averaged external potential (Vext) for CO2 in ALF
# using the warp GPU backend (LJ + smeared-Coulomb kernels).
#
# Runs phase1_vext_validation.py which:
#   1. Loads the ALF structure and CO2 EPM2 forcefield
#   2. Builds Vext on a 3D grid via warp kernels (use_warp=True)
#   3. Saves vext_avg.npy + a diagnostic slice plot to results/alf_co2/vext/
#
# Submit:
#   sbatch scripts/slurm/01_alf_co2_vext_gpu.sh
#
# Or via sync script:
#   bash scripts/slurm/sync_yembal.sh submit vext

set -e

PROJECT_ROOT="/home/conrard/porecdft"
CONDA_ENV="jax"
MODULE="nvhpc/26.3"

cd "$PROJECT_ROOT"
mkdir -p logs results/alf_co2/vext

echo "=================================================="
echo "  Job ID : $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo "  GPU    : $CUDA_VISIBLE_DEVICES"
echo "  Start  : $(date)"
echo "=================================================="

module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -c "import warp as wp; wp.init(); print('Warp devices:', wp.get_preferred_device())"
python -c "import jax; print('JAX devices:', jax.devices())"

python applications/alf_co2/notebooks/phase1_vext_validation.py \
    --use-warp \
    --output-dir results/alf_co2/vext \
    2>&1 | tee logs/vext_gpu_${SLURM_JOB_ID}.log

echo "=================================================="
echo "  Vext GPU build complete: $(date)"
echo "  Results in results/alf_co2/vext/"
echo "=================================================="
