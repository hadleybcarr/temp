#!/bin/bash
#SBATCH --job-name=compare_detectors
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                  # request 1 GPU
#SBATCH --mem=32G                     # RAM (Mask R-CNN + YOLO both loaded)
#SBATCH --cpus-per-task=4             # for OpenCV video decoding
#SBATCH --time=04:00:00               # wall-time – adjust if needed
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module load python/3.11.0s
module load cuda/11.8.0               # match your torch build
module load cudnn/8.7.0               # match your torch build

# Activate your virtual-env / conda env
# Option A – virtualenv:
source myenv/bin/activate
# Option B – conda (uncomment and adjust):
# module load miniconda3/23.11.0
# conda activate parking

# Make sure logs dir exists
mkdir -p logs

# ── Sanity check ───────────────────────────────────────────────────────────────
echo "-----------------------------------------------"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Python      : $(which python3)"
echo "PyTorch GPU : $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')"
echo "-----------------------------------------------"

# ── Run ────────────────────────────────────────────────────────────────────────
python3 compare_detectors.py \
    --sample-every 5 \
    --plot-out     detector_comparison.png
    # add --skip-spots  if you don't have infer_spots.py
    # add --no-run      to only regenerate the chart from existing pickles
    # add --variants yolov8 yolo11   to run a subset