#!/bin/bash
#SBATCH -J maskrcnn-spots
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o logs/maskrcnn-%j.out
#SBATCH -e logs/maskrcnn-%j.err
#Run: sbatch run_yolo_mask.sh


PROJECT=/oscar/data/class/csci1430/students/hbcarr/parking


mkdir -p logs

source myenv/bin/activate
#module load anaconda3
python -c "import cv2; print('cv2:', cv2.__version__)"


set -euo pipefail

echo "Running on $(hostname)"
#nvidia-smi || true

python3 mask_yolo_cnn.py \
    --out-dir      "$PROJECT/detection_cache_mask_yolo_rcnn" \
    --cameras-json "$PROJECT/videos_by_camera.json" \
    --n-samples-per-video 100 \
    --overwrite