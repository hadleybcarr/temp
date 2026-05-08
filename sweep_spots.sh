#!/bin/bash
#SBATCH -J spot_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o logs/sweep-%j.out
#SBATCH -e logs/sweep-%j.err

set -euo pipefail

# --- environment ---
module load python/3.9.16s-x3wdtvt 2>/dev/null || module load python/3.9 || true
module load cuda/11.8 2>/dev/null || true

source /oscar/home/hbcarr/temp/myenv/bin/activate

# --- paths ---
PROJECT_DIR=/oscar/home/hbcarr/temp
CACHE_DIR=/oscar/data/class/csci1430/students/hbcarr/parking/caches/mask_yolo_world
OUT_DIR=${PROJECT_DIR}/sweep_results

mkdir -p "${OUT_DIR}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"


VARIANT=$(basename "${CACHE_DIR}")
SWEEP_OUT="param_compare_${VARIANT}.json"

echo "host:      $(hostname)"
echo "python:    $(which python3)"
echo "cache dir: ${CACHE_DIR}"
echo "out dir:   ${OUT_DIR}/${SWEEP_OUT}"

python3 infer_spots.py \
    --cache-dir "${CACHE_DIR}" \
    --out-dir   "${OUT_DIR}" \
    --sweep-out "${SWEEP_OUT}" \
    --sweep