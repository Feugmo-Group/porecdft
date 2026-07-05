#!/bin/bash
#SBATCH --job-name=porecdft_compare_vext
#SBATCH --partition=sbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/compare_vext_%j.out
#SBATCH --error=logs/compare_vext_%j.err

# Validate CPU vs GPU Vext agreement for H2/COF (Morse + LJ composite potential).
# Runs compare_vext.py which builds Vext with both CPU-numpy and warp-GPU backends
# and asserts np.allclose(vext_cpu, vext_gpu, atol=1.0, rtol=1e-3). Fails loudly if mismatch.
#
# This is the regression test for the Morse warp kernel fix (fluid_params=None,
# include_species filtering, dict-or-dataclass _get helper) on the warp branch.
#
# Submit:
#   sbatch scripts/slurm/02_compare_vext_gpu.sh
#
# Or via sync script:
#   bash scripts/slurm/sync_yembal.sh submit compare

set -e

PROJECT_ROOT="/home/conrard/porecdft"
CONDA_ENV="jax"
MODULE="nvhpc/26.3"

cd "$PROJECT_ROOT"
mkdir -p logs applications/h2_cof/results applications/h2_cof/figures

echo "=================================================="
echo "  Job ID : $SLURM_JOB_ID  |  Node: $SLURM_NODELIST"
echo "  GPU    : $CUDA_VISIBLE_DEVICES"
echo "  Start  : $(date)"
echo "=================================================="

module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -c "import warp as wp; wp.init(); print('Warp devices:', wp.get_preferred_device())"

python applications/h2_cof/notebooks/compare_vext.py \
    2>&1 | tee logs/compare_vext_${SLURM_JOB_ID}.log

echo "=================================================="
echo "  CPU vs GPU Vext comparison complete: $(date)"
echo "=================================================="
