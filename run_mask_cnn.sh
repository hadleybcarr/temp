#!/bin/bash
#SBATCH -J maskrcnn-spots
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o logs/maskrcnn-%j.out
#SBATCH -e logs/maskrcnn-%j.err

# ---- adjust these for your setup ----
PROJECT=/oscar/data/class/csci1430/students/hbcarr/parking
# -------------------------------------

set -euo pipefail
mkdir -p logs

module load anaconda3
source ~/.local/bin/env

echo "Running on $(hostname)"
nvidia-smi || true

cd "$PROJECT"

python infer_maskrcnn.py \
    --videos-dir   "$PROJECT/videos" \
    --out-dir      "$PROJECT/detection_cache_maskrcnn" \
    --cameras-json "$PROJECT/videos_by_camera.json" \
    --n-samples-per-video 100