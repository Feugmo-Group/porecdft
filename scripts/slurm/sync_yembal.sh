#!/usr/bin/env bash
# Sync porecdft code with yemba_local GPU cluster and manage SLURM jobs.
#
# Usage:
#   bash scripts/slurm/sync_yembal.sh push             # push code to cluster
#   bash scripts/slurm/sync_yembal.sh pull             # pull results back
#   bash scripts/slurm/sync_yembal.sh submit vext      # submit Vext GPU job (ALF/CO2)
#   bash scripts/slurm/sync_yembal.sh submit compare   # submit CPU vs GPU compare_vext
#   bash scripts/slurm/sync_yembal.sh status           # show SLURM queue

set -e

REMOTE="yemba_local"
REMOTE_DIR="/home/conrard/porecdft"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

usage() {
    echo "Usage: $0 {push|pull|submit vext|submit compare|status}"
    exit 1
}

case "$1" in

push)
    echo "── Pushing code to ${REMOTE}:${REMOTE_DIR} ──"
    rsync -avz --progress \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='results/' \
        --exclude='graphify-out/' \
        --exclude='*.egg-info' \
        --exclude='.venv' \
        --exclude='paper_old/' \
        --exclude='Presentation/' \
        "${LOCAL_DIR}/" \
        "${REMOTE}:${REMOTE_DIR}/"
    echo "── Installing package on remote ──"
    ssh "${REMOTE}" "cd ${REMOTE_DIR} && pip install -e . --quiet"
    echo "Done. Code synced and package installed."
    ;;

pull)
    echo "── Pulling results from ${REMOTE}:${REMOTE_DIR}/results/ ──"
    rsync -avz --progress \
        "${REMOTE}:${REMOTE_DIR}/results/" \
        "${LOCAL_DIR}/results/"
    echo "Done. Results synced."
    ;;

submit)
    case "$2" in
    vext)
        echo "── Submitting ALF/CO2 Vext GPU build job ──"
        ssh "${REMOTE}" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch scripts/slurm/01_alf_co2_vext_gpu.sh"
        ;;
    compare)
        echo "── Submitting CPU vs GPU Vext comparison job ──"
        ssh "${REMOTE}" "cd ${REMOTE_DIR} && mkdir -p logs && sbatch scripts/slurm/02_compare_vext_gpu.sh"
        ;;
    *)
        usage
        ;;
    esac
    ;;

status)
    echo "── SLURM queue on ${REMOTE} ──"
    ssh "${REMOTE}" "squeue -u \$(whoami) --format='%.10i %.25j %.8T %.10M %.5D %R'"
    ;;

*)
    usage
    ;;
esac
