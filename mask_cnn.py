import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

videos_by_camera = {
    "thayer_angle_1": ["black_car_leaving_thayer.mp4", "car_leaving_thayer.mp4", "gray_pullout.mp4", "parallel_parking_process.mp4", "thayer_dark_gray_car.mp4", "thayer_leaving_parking.mp4", "thayer_leaving_parking.mp4", "two_car_thayer_parking.mp4", "two_cars_leave.mp4", "white_parallel_parking.mp4", "white_parallel_thayer.mp4"],
    "thayer_angle_2": ["car_adjusting.mp4", "driving_to_parked.mp4"],
    "thayer_angle_3": ["gray_car_pullin_thayer.mp4",],
    "thayer_angle_4": ["outside_of_bounds.mp4"],
    "thayer_angle_5": ["parallel_park.mp4",],
    "thayer_angle_6": ["parked_to_parking.mp4"],
    "pickup_angle_1": ["cars_leave_cars_park.mp4", "cars_staying_and_leaving.mp4"],
    "pickup_angle_2": ["cars_leaving.mp4", ],
    "pickup_angle_3": ["gray_car_leaving.mp4"],
    "pickup_angle_4": ["multi_car.mp4", "multicar_movement.mp4", "one_car_pullout.mp4", "one_park_one_leave.mp4", "parallel_park_pullup.mp4", "parking_and_leaving.mp4", "several_car_movement.mp4", "several_cars_leave.mp4", "two_cars_leave.mp4", ],
    "pickup_angle_5": ["two_cars_leave.mp4","white_car_pullup.mp4" ]
}

VEHICLE_CLASSES_TORCHVISION = {3, 4, 6, 8}

SCORE_THRESH = 0.5    # detection confidence
MASK_THRESH = 0.5     # threshold to binarize the soft mask
MIN_MASK_PIXELS = 100 # drop tiny masks (noise)


def load_model():
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def frame_to_tensor(frame_bgr):
    """BGR uint8 (H,W,3) -> RGB float [0,1] tensor (3,H,W) on the active device."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    if torch.cuda.is_available():
        t = t.cuda()
    return t


def extract_detections(output):
    """
    From a single Mask R-CNN output dict, return a list of plain Python dicts
    with the bottom-of-mask anchor and bbox info. We deliberately do NOT keep
    the masks themselves around -- only the summary stats -- so the cache stays
    small and we don't hit OOM the way the YOLO pass did.
    """
    detections = []
    boxes  = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks  = output["masks"].cpu().numpy()  # (N, 1, H, W) float

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < SCORE_THRESH:
            continue
        if int(label) not in VEHICLE_CLASSES_TORCHVISION:
            continue

        binary = mask[0] > MASK_THRESH
        area = int(binary.sum())
        if area < MIN_MASK_PIXELS:
            continue

        ys, xs = np.where(binary)
        ground_x = float(np.median(xs))   # robust to a side mirror sticking out
        ground_y = float(ys.max())        # bottom-most pixel of the car

        x1, y1, x2, y2 = box.tolist()
        detections.append({
            # primary anchor for clustering
            "ground_x": ground_x,
            "ground_y": ground_y,
            # legacy fields so this cache slots into the existing DBSCAN code
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "w":  x2 - x1,
            "h":  y2 - y1,
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "score": float(score),
            "label": int(label),
            "mask_area": area,
        })
    return detections


def sample_frame_indices(total_frames, n_samples):
    if total_frames <= 0:
        return []
    if total_frames <= n_samples:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n_samples, dtype=int).tolist()


def process_video(video_path, model, n_samples):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total, n_samples)

    detections = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        t = frame_to_tensor(frame)
        with torch.no_grad():
            out = model([t])[0]

        per_frame = extract_detections(out)
        for d in per_frame:
            d["frame_idx"] = int(idx)
            d["video"] = video_path.name
        detections.extend(per_frame)

        # Free GPU memory aggressively -- masks are big.
        del out, t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cap.release()
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to write per-camera pickle caches")
    parser.add_argument("--cameras-json", type=Path, required=True,
                        help="JSON mapping camera_id -> list of video filenames")
    parser.add_argument("--n-samples-per-video", type=int, default=100,
                        help="Frames sampled per video (evenly spaced)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if a camera's cache file already exists")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)


    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading Mask R-CNN (torchvision, COCO weights)...")
    model = load_model()

    for camera_id, video_names in videos_by_camera.items():
        out_path = args.out_dir / f"{camera_id}.pkl"
        if out_path.exists() and not args.overwrite:
            print(f"[{camera_id}] cache exists, skipping (use --overwrite to redo)")
            continue

        print(f"[{camera_id}] processing {len(video_names)} video(s)")
        all_detections = []
        for vn in video_names:
            video_path = "/oscar/data/class/csci1430/students/hbcarr/parking/videos/"+vn
            if not video_path.exists():
                print(f"  ! missing: {video_path}")
                continue
            dets = process_video(video_path, model, args.n_samples_per_video)
            print(f"  {vn}: {len(dets)} vehicle detections")
            all_detections.extend(dets)

        with open(out_path, "wb") as f:
            pickle.dump(all_detections, f)
        print(f"[{camera_id}] saved {len(all_detections)} detections -> {out_path}")


if __name__ == "__main__":
    main()