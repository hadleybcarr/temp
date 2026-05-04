"""
Run every detector variant on the same videos and produce a comparison chart.

Variants compared:
    - yolov8         (yolov8m)
    - yolo11         (yolo11m)
    - yolo_world     (yolov8s-worldv2)
    - mask_yolo11    (Mask R-CNN fused with yolo11m)
    - mask_yolo_world(Mask R-CNN fused with yolov8s-worldv2)

For each variant the script:
    1. Skips models whose pickle already exists at detections/<variant>/<cam>.pkl
       (so you can drop in pickles you already have -- just put them in the
        right subfolder with the matching filename and they'll be picked up).
    2. Otherwise runs the detector on every video listed in --videos-json
       and saves a pickle in the canonical schema (x1/y1/x2/y2/cx/cy/w/h/
       ground_x/ground_y/score/source/frame_idx/video).
    3. Optionally runs infer_spots.py on each pickle to also compare the
       *downstream spot count* per variant, not just raw detections.
    4. Builds a grouped bar chart: rows=cameras, bars=variants, height=count.

Usage (on Oscar):
    python compare_detectors.py \\
        --videos-json   videos_by_camera.json \\
        --videos-dir    /oscar/data/.../parking/videos \\
        --pickle-root   detections \\
        --plot-out      detector_comparison.png \\
        --skip-spots                # only graph raw-detection counts
"""

import argparse
import json
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


videos_by_camera = {
    "thayer_angle_1": ["black_car_leaving_thayer.mp4", "car_leaving_thayer.mp4", "gray_pullout.mp4", "two_car_thayer_parking.mp4", "white_parallel_parking.mp4", "white_parallel_thayer.mp4"],
    "thayer_angle_2": ["car_adjusting.mp4", "driving_to_parked.mp4"],
    "thayer_angle_3": ["gray_car_pullin_thayer.mp4", "parallel_parking_process.mp4", "thayer_dark_gray_car.mp4", "thayer_leaving_parking.mp4"],
    "thayer_angle_4": ["outside_of_bounds.mp4"],
    "thayer_angle_5": ["parallel_park.mp4"],
    "thayer_angle_6": ["parked_to_parking.mp4"],
    "pickup_angle_1": ["cars_leave_cars_park.mp4", "cars_staying_and_leaving.mp4"],
    "pickup_angle_2": ["cars_leaving.mp4"],
    "pickup_angle_3": ["gray_car_leaving.mp4"],
    "pickup_angle_4": ["multi_car.mp4", "multicar_movement.mp4", "one_car_pullout.mp4", "one_park_one_leave.mp4", "parallel_park_pullup.mp4", "parking_and_leaving.mp4", "several_car_movement.mp4", "several_cars_leave.mp4"],
    "pickup_angle_5": ["two_cars_leave.mp4","white_car_pullup.mp4" ]
}





# ---------- model registry ----------

MODELS = {
    "yolov8":          {"kind": "yolo",  "weights": "yolov8m.pt"},
    "yolo11":          {"kind": "yolo",  "weights": "yolo11m.pt"},
    "yolo_world":      {"kind": "yolo",  "weights": "yolov8s-worldv2.pt"},
    "maskrcnn":       {"kind": "maskrcnn", "weights": None},
    "mask_yolo_rcnn":     {"kind": "fused", "yolo_weights": "yolo11m.pt"},
    "mask_yolo_world": {"kind": "fused", "yolo_weights": "yolov8s-worldv2.pt"},
}

YOLO_VEHICLE_CLASSES   = [2, 3, 5, 7]    # car, motorcycle, bus, truck (0-indexed COCO)
MASK_VEHICLE_CLASSES   = [3, 4, 6, 8]    # car, motorcycle, bus, truck (1-indexed COCO)
MASK_SCORE_THRESH      = 0.5
YOLO_CONF              = 0.20
MATCH_IOU              = 0.40            # used for fused (yolo + mask) matching


# ---------- detection schema ----------

def _detection(x1, y1, x2, y2, score, source, frame_idx, video,
               ground_x=None, ground_y=None):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    return {
        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
        "cx": cx, "cy": cy, "w": w, "h": h,
        "ground_x": float(ground_x if ground_x is not None else cx),
        "ground_y": float(ground_y if ground_y is not None else y2),
        "score":    float(score),
        "source":   source,
        "frame_idx": int(frame_idx),
        "video":    video,
    }


# ---------- runners ----------

def run_yolo(model, frame, frame_idx, video):
    out = []
    res = model(frame, imgsz=1280, conf=YOLO_CONF, classes=YOLO_VEHICLE_CLASSES,
                verbose=False)
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            score = float(b.conf[0])
            out.append(_detection(x1, y1, x2, y2, score, "yolo_only",
                                  frame_idx, video))
    return out


def run_maskrcnn(model, frame, frame_idx, video, device):
    import torch
    from torchvision.transforms.functional import to_tensor
    img = to_tensor(frame[..., ::-1].copy()).to(device)
    with torch.no_grad():
        pred = model([img])[0]
    out = []
    for box, label, score, mask in zip(pred["boxes"], pred["labels"],
                                       pred["scores"], pred["masks"]):
        if label.item() not in MASK_VEHICLE_CLASSES:
            continue
        if score.item() < MASK_SCORE_THRESH:
            continue
        x1, y1, x2, y2 = box.cpu().numpy().tolist()
        # bottom-of-mask ground anchor (perspective-robust)
        m = mask[0].cpu().numpy() > 0.5
        ys, xs = np.where(m)
        if len(ys) == 0:
            gx, gy = (x1 + x2) / 2, y2
        else:
            gy = float(ys.max())
            gx = float(xs[ys == ys.max()].mean())
        out.append(_detection(x1, y1, x2, y2, float(score), "maskrcnn_only",
                              frame_idx, video, ground_x=gx, ground_y=gy))
    return out


def iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    bb = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (aa + bb - inter)


def fuse(yolo_dets, mask_dets):
    """Greedy IoU match. Prefer Mask R-CNN's ground anchor when matched."""
    fused = []
    used = set()
    for y in yolo_dets:
        best_j, best_i = -1, 0.0
        for j, m in enumerate(mask_dets):
            if j in used:
                continue
            ii = iou(y, m)
            if ii > best_i and ii >= MATCH_IOU:
                best_i, best_j = ii, j
        if best_j >= 0:
            m = mask_dets[best_j]
            used.add(best_j)
            fused.append(_detection(
                (y["x1"] + m["x1"]) / 2, (y["y1"] + m["y1"]) / 2,
                (y["x2"] + m["x2"]) / 2, (y["y2"] + m["y2"]) / 2,
                max(y["score"], m["score"]),
                "fused",
                y["frame_idx"], y["video"],
                ground_x=m["ground_x"], ground_y=m["ground_y"],
            ))
        else:
            fused.append({**y, "source": "yolo_only"})
    for j, m in enumerate(mask_dets):
        if j not in used:
            fused.append({**m, "source": "maskrcnn_only"})
    return fused


def run_video(variant, video_path, sample_every=5):
    print("Running model...")
    """Returns a list of detection dicts for the given video + variant."""
    spec = MODELS[variant]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ! could not open {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = list(range(0, total, sample_every))

    out = []

    if spec["kind"] == "yolo":
        from ultralytics import YOLO
        import torch
        model = YOLO(spec["weights"])
        if torch.cuda.is_available():
            model.to("cuda")
        for f_idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out.extend(run_yolo(model, frame, f_idx, video_path.name))

    elif spec["kind"] == "maskrcnn":
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mmodel = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device).eval()
        for f_idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            out.extend(run_maskrcnn(mmodel, frame, f_idx, video_path.name, device))

    elif spec["kind"] == "fused":
        from ultralytics import YOLO
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ymodel = YOLO(spec["yolo_weights"])
        if device == "cuda":
            ymodel.to("cuda")
        mmodel = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device).eval()
        for f_idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            yd = run_yolo(ymodel, frame, f_idx, video_path.name)
            md = run_maskrcnn(mmodel, frame, f_idx, video_path.name, device)
            out.extend(fuse(yd, md))

    cap.release()
    return out


# ---------- driver ----------

def ensure_pickles(args):
    print("Ensuring pickles...")
    """Run any (variant, camera) combination whose pickle is missing."""
    pickle_root = "/oscar/data/class/csci1430/students/hbcarr/parking/caches"
    variants = ["yolov8", "yolo11", "yolo_world", "maskrcnn", "mask_yolo_rcnn", "mask_yolo_world"]
    for variant in variants:
        out_dir = pickle_root / variant
        print(out_dir, "is output directory")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{variant}] cache dir: {out_dir}")
        for camera, videos in videos_by_camera.items():
            pkl = out_dir / f"{camera}.pkl"
            if pkl.exists():
                print(f"[{variant}/{camera}] cached -> {pkl}")
                continue
            print(f"[{variant}/{camera}] running on {len(videos)} video(s)")
            dets = []
            video_directory = Path("/oscar/data/class/csci1430/students/hbcarr/parking/videos")
            for v in videos:
                vp = video_directory / v
                if not vp.exists():
                    print(f"  ! missing video: {vp}")
                    continue
                dets.extend(run_video(variant, vp, sample_every=args.sample_every))
            with open(pkl, "wb") as f:
                pickle.dump(dets, f)
            print(f"  -> {len(dets)} detections written to {pkl}")


def summarize_pickles(args, videos_by_camera):
    """Walk the pickles and tabulate per-(camera, variant) detection counts."""
    table = defaultdict(dict)        # table[camera][variant] = n_detections
    confs = defaultdict(list)        # confs[variant]         = [scores ...]
    for variant in args.variants:
        for camera in videos_by_camera:
            pkl = Path("/oscar/data/class/csci1430/students/hbcarr/parking/caches") / variant / f"{camera}.pkl"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                dets = pickle.load(f)
            table[camera][variant] = len(dets)
            confs[variant].extend(d.get("score", 0.0) for d in dets)
    return table, confs


def run_infer_spots(args, videos_by_camera):
    """Optionally run infer_spots.py on each variant's pickle directory and
    return spot counts per (camera, variant)."""
    spots_table = defaultdict(dict)
    for variant in args.variants:
        cache_dir =  variant
        spots_json = Path(cache_dir +"/_spots.json")
        if not spots_json.exists():
            print(f"[{variant}] running infer_spots.py")
            subprocess.run([
                "python3", "infer_spots.py",
                "--cache-dir", str(cache_dir),
            ], check=False)
            # infer_spots writes to a file in CWD; pull it in if it exists
            for candidate in ("mask_spots_by_cam.json", "spots_by_camera.json"):
                if Path(candidate).exists():
                    Path(candidate).rename(spots_json)
                    break
        if not spots_json.exists():
            print(f"  ! couldn't locate spots json for {variant}")
            continue
        data = json.loads(spots_json.read_text())
        # support both flat and nested schemas
        cams = data.get("cameras", data)
        for camera in videos_by_camera:
            entries = cams.get(camera, [])
            if isinstance(entries, dict):
                entries = entries.get("spots", [])
            n_real = sum(1 for s in entries
                         if s.get("source") in ("clustered", "extrapolated"))
            spots_table[camera][variant] = n_real
    return spots_table


def make_plot(det_table, spot_table, variants, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cameras = sorted(det_table.keys())
    n_cams = len(cameras)
    n_var = len(variants)
    bar_w = 0.8 / max(n_var, 1)
    x = np.arange(n_cams)

    fig, axes = plt.subplots(2 if spot_table else 1, 1,
                             figsize=(max(8, 1.4 * n_cams), 8 if spot_table else 5),
                             squeeze=False)
    palette = plt.cm.tab10(np.linspace(0, 1, n_var))

    # --- detections per camera ---
    ax = axes[0, 0]
    for j, v in enumerate(variants):
        ys = [det_table[c].get(v, 0) for c in cameras]
        ax.bar(x + j * bar_w - 0.4 + bar_w / 2, ys, bar_w,
               label=v, color=palette[j])
    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=30, ha="right")
    ax.set_ylabel("# raw detections")
    ax.set_title("Detection count per camera, by detector variant")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- inferred spots per camera (optional) ---
    if spot_table:
        ax = axes[1, 0]
        for j, v in enumerate(variants):
            ys = [spot_table.get(c, {}).get(v, 0) for c in cameras]
            ax.bar(x + j * bar_w - 0.4 + bar_w / 2, ys, bar_w,
                   label=v, color=palette[j])
        ax.set_xticks(x)
        ax.set_xticklabels(cameras, rotation=30, ha="right")
        ax.set_ylabel("# inferred spots")
        ax.set_title("Inferred parking spots per camera (after infer_spots.py)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote chart -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle-root", type=Path, default=Path("detections"),
                        help="Per-variant pickles live in <root>/<variant>/<camera>.pkl")
    parser.add_argument("--variants",    nargs="*", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Which variants to include (default: all)")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--plot-out",    type=Path, default=Path("detector_comparison.png"))
    parser.add_argument("--skip-spots",  action="store_true",
                        help="Skip the infer_spots step; only chart raw detections")
    parser.add_argument("--no-run",      action="store_true",
                        help="Don't run any new inference; only use existing pickles")
    args = parser.parse_args()

    '''python3 compare_detectors.py \
    --pickle-root detections \
    --plot-out    detector_comparison.png
'''


    if not args.no_run:
        ensure_pickles(args)

    det_table, confs = summarize_pickles(args, videos_by_camera)
    print("\nRaw detection counts:")
    for camera in sorted(det_table):
        line = f"  {camera:<25s}"
        for v in args.variants:
            n = det_table[camera].get(v, 0)
            line += f"  {v}={n:>5d}"
        print(line)

    spot_table = None
    if not args.skip_spots:
        spot_table = run_infer_spots(args, videos_by_camera)
        print("\nInferred spot counts:")
        for camera in sorted(spot_table):
            line = f"  {camera:<25s}"
            for v in args.variants:
                n = spot_table[camera].get(v, 0)
                line += f"  {v}={n:>4d}"
            print(line)

    make_plot(det_table, spot_table, args.variants, args.plot_out)


if __name__ == "__main__":
    main()