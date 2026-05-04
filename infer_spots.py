"""
Infer parking-spot geometry from the per-camera pickle caches written by
infer_maskrcnn.py.

Pipeline per camera:
    1. DBSCAN cluster the bottom-of-mask points (ground_x, ground_y).
       (Bottom-of-mask is perspective-robust; this is the main reason we
        switched from YOLO box centers.)
    2. For each cluster, compute summary stats: median ground point,
       median car footprint, observation count, longest dwell run.
    3. Reject clusters that look like through-traffic (low dwell count).
    4. Split surviving clusters into two rows (one per side of the street).
    5. For each row, fit a curb line and extrapolate missing spots into
       car-sized gaps.
    6. Write spots_by_camera.json with the schema your downstream
       crop-and-label pass expects.


Usage:
     python3 infer_spots.py --cache-dir mask_cnn
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import open_clip
from PIL import Image
import torch

FRAME_SIZE = 3000 * 3000

# ---------- helpers ----------

def to_native(x):
    """Recursively cast numpy scalars/arrays to plain Python so json.dump works."""
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_TEXT = None
_CLIP_PROMPTS = [
    "a photo of a parked car",
    "a photo of a trash bin or dumpster",
    "a photo of a road sign or A-frame sign",
    "a photo of construction equipment",
    "a photo of an empty parking spot",
]

def _ensure_clip():
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_TEXT
    if _CLIP_MODEL is not None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _CLIP_MODEL, _, _CLIP_PREPROCESS = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _CLIP_MODEL = _CLIP_MODEL.to(device).eval()
    _CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-B-32")
    _CLIP_TEXT = _CLIP_TOKENIZER(_CLIP_PROMPTS).to(device)

def is_car(cluster, videos_dir, car_prob_thresh=0.45):
    """Crop the highest-confidence detection in this cluster, ask CLIP if it's a car."""
    _ensure_clip()
    members = cluster.get("_raw_points", [])
    if not members:
        return True   # nothing to check, don't reject
    m = max(members, key=lambda d: d.get("confidence", 0))

    cap = cv2.VideoCapture(str(Path(videos_dir) / m["video"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, m["frame_idx"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return True  

    print("m is", m)
    gx, gy = cluster["ground_x"], cluster["ground_y"]
    w, h   = cluster.get("w", 80), cluster.get("h", 80)
    bbox = (gx - w / 2.0, gy - h, gx + w / 2.0, gy)
    x1, y1, x2, y2 = bbox

    H, W = frame.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return True

    img = _CLIP_PREPROCESS(Image.fromarray(crop[..., ::-1])).unsqueeze(0)
    img = img.to(_CLIP_TEXT.device)
    with torch.no_grad():
        if hasattr(_CLIP_MODEL, "encode_image"):   # open_clip path
            image_features = _CLIP_MODEL.encode_image(img)
            text_features  = _CLIP_MODEL.encode_text(_CLIP_TEXT)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
        else:                                      # openai-clip path
            logits, _ = _CLIP_MODEL(img, _CLIP_TEXT)
    car_prob = logits.softmax(dim=-1)[0, 0].item()
    return car_prob >= car_prob_thresh

def max_consecutive_run(frames, gap_tolerance=2):
    """
    Longest run of "present" frames, allowing up to `gap_tolerance` consecutive
    misses before resetting. With sampled-frame data this counts how many
    consecutive *sampled* frames a cluster appears in.
    """
    if not frames:
        return 0
    frames = sorted(set(int(f) for f in frames))
    longest = current = 1
    for prev, nxt in zip(frames, frames[1:]):
        if nxt - prev <= gap_tolerance + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest

def detect_road(detections):
    eps = 20
    max_visit_dwell = 3
    by_video = defaultdict(list)
    for d in detections:
        by_video[d["video"]].append(d)

    moving_points = []
    for video, dets in by_video.items():
        dets.sort(key=lambda d:d["frame_idx"])
        for prev, cur in zip(dets, dets[1:]):
            df = cur["frame_idx"] - prev["frame_idx"]
        if df == 0: 
            continue


def cluster_detections(detections, eps, min_samples):
    """
    DBSCAN on (ground_x, ground_y). Returns a dict: cluster_id -> stats.
    Cluster -1 (noise) is dropped.
    """
    size_min_ratio = 0.5
    size_max_ratio = 2.0
    visit_gap_frames = 20.0
    min_visit_dwell = 2
    max_visit_velocity = 4.0
    max_intra_visit_std = 30.0
    min_visits = 1

    if len(detections) < min_samples:
        return {}
    
    areas = np.array([d["w"] * d["h"] for d in detections], dtype = float)
    median_area = float(np.median(areas))
    lo, hi = size_min_ratio * median_area, size_max_ratio * median_area
    keep = (areas >= lo) & (areas <= hi)
    detections = [d for d,k in zip(detections, keep) if k]
    
    if len(detections) < min_samples:
        return {}
    
    #print("Point structure is", detections[0])
    points = np.array([[d["ground_x"], d["ground_y"]] for d in detections])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    clusters = {}
    rejects = []
    
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        members = [d for d, l in zip(detections, labels) if l == lbl]

        by_video = defaultdict(list)
        for m in members:
            by_video[m.get("video", "_unknown")].append(m)

        visits = []
        for video, dets in by_video.items():
            dets.sort(key=lambda d: d.get("frame_idx", 0))
            run = [dets[0]]
            for d in dets[1:]:
                if d["frame_idx"] - run[-1]["frame_idx"] <= visit_gap_frames:
                    run.append(d)
                else:
                    visits.append(run)
                    run = [d]
            visits.append(run)

        valid_visits = []
        velocities  = []
        max_dwell   = 0

        for vdets in visits:
            if len(vdets) < 2:
                continue

            xs = np.array([d["ground_x"]   for d in vdets])
            ys = np.array([d["ground_y"]   for d in vdets])
            fs = np.array([d["frame_idx"]  for d in vdets])

            dwell = int(fs[-1] - fs[0])
            max_dwell = max(max_dwell, dwell)

            intra_std = float(np.sqrt(xs.var() + ys.var()))

            df = np.maximum(np.diff(fs), 1)              # guard divide-by-zero
            speeds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / df
            mean_v = float(speeds.mean())
            velocities.append(mean_v)

            if (dwell      >= min_visit_dwell   and
                mean_v     <= max_visit_velocity and
                intra_std  <= max_intra_visit_std):
                valid_visits.append(vdets)
        
        if len(valid_visits) < min_visits:
            rejects.append((lbl, len(members), len(visits),
                            len(valid_visits), max_dwell,
                            "too_few_parked_visits"))
            continue
        
        gx = np.array([m["ground_x"] for m in members])
        gy = np.array([m["ground_y"] for m in members])
        ws = np.array([m["w"] for m in members])
        hs = np.array([m["h"] for m in members])
        frames = [m["frame_idx"] for m in members]
        videos = set(m["video"] for m in members)

        clusters[int(lbl)] = {
            "ground_x": float(np.median(gx)),
            "ground_y": float(np.median(gy)),
            "w": float(np.median(ws)),
            "h": float(np.median(hs)),
            "n_observations": int(len(members)),
            "n_videos": int(len(videos)),
            "max_dwell_frames": int(max_consecutive_run(frames)),
            "_raw_points": members,
        }
    return clusters

def classify_parking(clusters, min_dwell=5, min_observations=30):
    """
    Tag each cluster as 'clustered' (parked) or 'rejected_road' (traffic).
    Heuristic: a real parking spot has lots of observations AND those
    observations span multiple sampled frames consecutively.
    """
    for cid, c in clusters.items():
        max_dwell_frames = c["max_dwell_frames"]
        max_observations = c["n_observations"]
        is_parked = (c["max_dwell_frames"] >= 1
                     and c["n_observations"] >= 0)
        print(max_dwell_frames, "and", max_observations)
        c["source"] = "clustered" if is_parked else "rejected_road"

    parked   = [c for c in clusters.values() if c["source"] == "clustered"]
    rejected = [c for c in clusters.values() if c["source"] == "rejected_road"]
    print(f"  [diag] classify -> {len(parked)} parked, {len(rejected)} rejected")
    return clusters


def split_into_rows(parked):
    """
    Given the list of parked-cluster stats, fit a single line through all of
    them, then KMeans (k=2) on the perpendicular distance to that line to get
    two rows -- one per side of the street.

    Returns:
        along_unit:    unit vector along the street
        perp_unit:     unit vector perpendicular to the street
        origin:        a reference point (centroid)
        row_labels:    list of 0/1 per cluster, same order as `parked`
    """
    pts = np.array([[c["ground_x"], c["ground_y"]] for c in parked])
    if len(pts) < 10:
        # too few to split; treat them all as row 0
        return None, None, pts.mean(axis=0), np.zeros(len(pts), dtype=int)

    # principal direction via covariance eigen-decomposition
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    along_unit = eigvecs[:, np.argmax(eigvals)]
    perp_unit = np.array([-along_unit[1], along_unit[0]])

    median_w = float(np.median([c["w"] for c in parked]))
    median_h = float(np.median([c["h"] for c in parked]))
    wide = median_w >= median_h
    horizontal = abs(along_unit[0] <= abs(along_unit[1]))
    if wide != horizontal:
        along_unit, perp_unit = perp_unit, along_unit

    perp_proj = centered @ perp_unit  # signed distance from the centroid line
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(perp_proj.reshape(-1, 1))
    row_labels = km.labels_

    return along_unit, perp_unit, pts.mean(axis=0), row_labels

def fit_curb_line(pts, n_iters=200, inlier_thresh=6.0, rng=None):
    """RANSAC line fit. Pick 2 random points, count inliers, keep best model."""
    pts = pts.astype(np.float32)
    if len(pts) < 3:
        # fall back to OLS
        x0, y0 = pts.mean(axis=0)
        vx, vy = (pts[-1] - pts[0]) / (np.linalg.norm(pts[-1] - pts[0]) + 1e-9)
        return np.array([vx, vy]), np.array([-vy, vx]), np.array([x0, y0])

    rng = np.random.default_rng(0) if rng is None else rng
    best_inliers, best_model = -1, None
    for _ in range(n_iters):
        i, j = rng.choice(len(pts), size=2, replace=False)
        p1, p2 = pts[i], pts[j]
        v = p2 - p1
        if np.linalg.norm(v) < 1e-6:
            continue
        v = v / np.linalg.norm(v)
        perp = np.array([-v[1], v[0]])
        d = np.abs((pts - p1) @ perp)
        n_in = int((d < inlier_thresh).sum())
        if n_in > best_inliers:
            best_inliers, best_model = n_in, (v, perp, p1)

    v, perp, p1 = best_model
    # refine with least-squares on the inlier set
    d = np.abs((pts - p1) @ perp)
    inliers = pts[d < inlier_thresh]
    vx, vy, x0, y0 = cv2.fitLine(inliers, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([vx, vy]), np.array([-vy, vx]), np.array([x0, y0])

def process_row(row_clusters, camera_id, row_id, global_along=None, gap_tolerance=0.4,
                max_perp_offset=6.0):
    """
    Returns (spots, line_info). Always a 2-tuple.
    Spots are snapped onto the fitted curb line; clustered spots may keep a
    small clamped perpendicular offset.
    """
    max_angle_deg = 5
    n_endpoints = 2 

    if len(row_clusters) < 3:
        spots = []
        for i, c in enumerate(row_clusters):
            spots.append(_make_spot(
                gx=c["ground_x"], gy=c["ground_y"],
                w=c["w"], h=c["h"],
                along=np.array([1.0, 0.0]),
                camera_id=camera_id, row_id=row_id, idx=i,
                source=c["source"],
                n_obs=c["n_observations"],
                dwell=c["max_dwell_frames"],
            ))
        return spots, None
 
    widths  = np.array([c["w"] for c in row_clusters])
    heights = np.array([c["h"] for c in row_clusters])
    pts = np.array([[c["ground_x"], c["ground_y"]] for c in row_clusters])

    survivors = list(row_clusters)
    if len(survivors) >= 5:
        pts0 = np.array([[c["ground_x"], c["ground_y"]] for c in survivors])
        a0, p0, o0 = fit_curb_line(pts0)
        worst = int(np.argmax(np.abs((pts0 - o0) @ p0)))
        survivors.pop(worst)
    
    pts = np.array([[c["ground_x"], c["ground_y"]] for c in survivors])
    centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    along_pre = vh[0]
    order = np.argsort(centered @ along_pre)
    sorted_proj = [survivors[i] for i in order]
 

    keep = list(sorted_proj)
    endpoint_idxs = list(range(min(n_endpoints, len(sorted_proj)))) + \
                    list(range(-min(n_endpoints, len(sorted_proj)), 0))
    for idx in endpoint_idxs:
        c = sorted_proj[idx]
        if not is_car(c, Path("videos")):
            keep = [k for k in keep if k is not c]
        if len(keep) < 3:
            return [], None
        
    pts     = np.array([[c["ground_x"], c["ground_y"]] for c in keep])
    widths  = np.array([c["w"] for c in keep])
    heights = np.array([c["h"] for c in keep])
    along, perp, origin = fit_curb_line(pts)
        
    if global_along is not None:
        cos_sim = abs(float(np.dot(along, global_along)))
        cos_sim = min(cos_sim, 1.0)
        deviation = np.degrees(np.arccos(cos_sim))
        if deviation > max_angle_deg:
            along = global_along
            perp = np.array([-along[1], along[0]])
            origin = pts.mean(axis=0)

    rel = pts - origin
    proj = rel @ along
    perp_offsets = rel @ perp
    sort_idx = np.argsort(proj)
    proj = proj[sort_idx] 
    perp_offsets = perp_offsets[sort_idx]
    keep_sorted = [keep[i] for i in sort_idx] 

    spot_w = float(np.median(widths))
    spot_h = float(np.median(heights))
 
    spots = []
    next_idx = 0

    # 5a. emit clustered spots, snapped onto the line (with small perp offset)
    for k, idx in enumerate(keep_sorted):
        perp_offset = float(np.clip(perp_offsets[k], -max_perp_offset, max_perp_offset))
        gx_line, gy_line = origin + proj[k] * along
        gx = gx_line + perp_offset * perp[0]
        gy = gy_line + perp_offset * perp[1]
        spots.append(_make_spot(
            gx=gx, gy=gy, w=spot_w, h=spot_h, along=along,
            camera_id=camera_id, row_id=row_id, idx=next_idx,
            source=c["source"],
            n_obs=c["n_observations"],
            dwell=c["max_dwell_frames"],
        ))
        next_idx += 1
 
    # 5b. extrapolate into car-sized gaps
    for i in range(len(proj) - 1):
        gap = proj[i + 1] - proj[i]
        n_extra = int(round(gap / spot_w)) - 1
        if n_extra <= 0:
            continue
        if gap < n_extra * spot_w * (1 - gap_tolerance):
            continue
        for k in range(1, n_extra + 1):
            t = proj[i] + k * (gap / (n_extra + 1))
            gx, gy = origin + t * along
            spots.append(_make_spot(
                gx=gx, gy=gy, w=spot_w, h=spot_h, along=along,
                camera_id=camera_id, row_id=row_id, idx=next_idx,
                source="extrapolated",
            ))
            next_idx += 1
 
    line_info = {
        "origin":             origin.tolist(),
        "along_unit":         along.tolist(),
        "perp_unit":          perp.tolist(),
        "median_spot_length": spot_w,
        "median_car_height":  spot_h,
        "n_real_spots":       int(len(row_clusters)),
        "n_extrapolated":     int(sum(1 for s in spots if s["source"] == "extrapolated")),
    }
    return spots, line_info


def _make_spot(gx, gy, w, h, along, camera_id, row_id, idx, source,
               n_obs=0, dwell=0):
    """
    Build a spot dict in the schema your downstream pipeline expects.

    The bbox is anchored at the bottom (ground_y) and extends UP by `h`,
    centered horizontally on `ground_x`. That mirrors how a parked car
    actually sits in image space.
    """
    
    if (w * h) > FRAME_SIZE:
        return 
    
    x1, y1 = gx - w / 2, gy - h
    x2, y2 = gx + w / 2, gy
    return {
        "box": [float(x1), float(y1), float(x2), float(y2)],
        "ground_x": float(gx),
        "ground_y": float(gy),
        "row": int(row_id),
        "source": source,
        #"n_observations": int(cluster["n_observations"]),
        #"max_dwell_frames": int(cluster["max_dwell_frames"]),
    }


def overlap_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


def deduplicate_spots(spots, iou_threshold=0.2):
    """
    If a 'rejected_road' spot overlaps a clustered or extrapolated spot,
    drop the rejected one. 
    """
    real = [s for s in spots if s["source"] in ("clustered", "extrapolated", "manual")]
    rejected = [s for s in spots if s["source"] == "rejected_road"]

    kept_rejected = []
    for r in rejected:
        if any(overlap_iou(r["box"], s["box"]) > iou_threshold for s in real):
            continue   # suppressed by a real spot
        kept_rejected.append(r)

    return real + kept_rejected


def infer_spots_for_camera(detections, eps, min_samples, min_dwell, min_ob, camera_id):
    print(f"  [diag] {len(detections)} detections in")
    clusters = cluster_detections(detections, eps=eps, min_samples=min_samples)
    if not clusters:
        return [], {}, {"n_clusters": 0}

    clusters = classify_parking(clusters,
                                min_dwell=1,
                                min_observations=5)

    parked = [c for c in clusters.values() if c["source"] == "clustered"]
    rejected = [c for c in clusters.values() if c["source"] == "rejected_road"]

    if not parked:
        return [], {}, {
            "n_clusters": len(clusters),
            "n_parked": 0,
            "n_rejected": len(rejected),
        }

    along_unit, perp_unit, origin, row_labels = split_into_rows(parked)

    spots = []
    line_infos = {}
    for row_id in (0, 1):
        in_row = [parked[i] for i in range(len(parked)) if row_labels[i] == row_id]

        row_spots, line_info = process_row(in_row, camera_id, row_id=row_id, global_along=along_unit)
        spots.extend(row_spots)
        if line_info is not None:
            line_infos[row_id] = line_info

    # also persist the rejected ones, tagged, in case you want to inspect them later
    #for c in rejected:
    #    spots.append(_make_spot(c, row_id=-1, source="rejected_road"))

    spots = deduplicate_spots(spots, iou_threshold=0.2)
    stats = {
        "n_clusters": len(clusters),
        "n_parked": len(parked),
        "n_rejected": len(rejected),
        "n_extrapolated": sum(1 for s in spots if s["source"] == "extrapolated"),
        "n_total_spots": sum(1 for s in spots if s["source"] != "rejected_road"),
    }

    return spots, line_infos, stats

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="Directory containing per-camera .pkl files")
    parser.add_argument("--lines-json", type=Path,
                        default=None,
                        help="Optional: also write per-row line geometry to this file")
    parser.add_argument("--eps", type=float, default=25,
                        help="DBSCAN eps in pixels (cluster radius)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="DBSCAN min_samples")
    parser.add_argument("--min-dwell", type=int, default=1,
                        help="A cluster must appear in this many consecutive sampled frames")
    parser.add_argument("--min-obs", type=int, default=3,
                        help="A cluster must have at least this many total observations")
    args = parser.parse_args()

    pickles = sorted(args.cache_dir.glob("*.pkl"))
    if not pickles:
        raise SystemExit(f"No .pkl files found in {args.cache_dir}")

    spots_by_camera = {}
    lines_by_camera = {}

    for pkl in pickles:
        camera_id = pkl.stem
        with open(pkl, "rb") as f:
            detections = pickle.load(f)

        spots, line_infos, stats = infer_spots_for_camera(
            detections,
            eps=args.eps,
            min_samples=args.min_samples,
            min_dwell=args.min_dwell,
            min_ob=args.min_obs,
            camera_id=camera_id
        )
        spots_by_camera[camera_id] = spots
        lines_by_camera[camera_id] = line_infos

        print(f"[{camera_id}] {len(detections)} detections -> "
              f"{stats.get('n_total_spots', 0)} spots "
              f"({stats.get('n_parked', 0)} clustered + "
              f"{stats.get('n_extrapolated', 0)} extrapolated, "
              f"{stats.get('n_rejected', 0)} rejected)")


    with open("mask_spots_by_cam.json", "w") as f:
        json.dump(to_native(spots_by_camera), f, indent=2)

    with open("lines_by_camera.json", "w") as f:
        json.dump(to_native(lines_by_camera), f, indent = 2)

    raw_spots = {}
    for camera, spots in spots_by_camera.items():
        if len(spots) > 0:
            raw_spots[camera] = []
            for spot in spots:
                raw_spots[camera].append([spot["box"]])
            
    #print(raw_spots)
    with open("raw_spots.json", "w") as f:
        json.dump(to_native(raw_spots), f, indent=2)

#TODO: add in road logic, continue to tweak hyper parameters, fix extrapolation

if __name__ == "__main__":
    main()

'''
def median_process_row(row_clusters, row_id, gap_tolerance=0.4):
    """
    Fit a curb line through this row's clusters, then look for car-sized gaps
    along the line and synthesize a new spot for each gap.

    Args:
        row_clusters:    list of cluster-stat dicts (one row's worth)
        row_id:          0 or 1, just for tagging
        gap_tolerance:   a gap >= (1 + gap_tolerance) * median_spot_length
                         counts as room for an extra spot

    Returns:
        spots:           list of finalized spot dicts (clustered + extrapolated)
        line_info:       dict describing the fitted curb line, or None if too few points
    """
    if len(row_clusters) < 3:
        # too few to fit a meaningful line; emit the spots as-is, no extrapolation
        spots = []
        for c in row_clusters:
            spots.append(_make_spot(c, row_id, source=c["source"]))
        return spots, None

    pts = np.array([[c["ground_x"], c["ground_y"]] for c in row_clusters])
    widths = np.array([c["w"] for c in row_clusters])
    heights = np.array([c["h"] for c in row_clusters])

    # principal direction of THIS row (more accurate than the global one)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    along = eigvecs[:, np.argmax(eigvals)]
    perp = np.array([-along[1], along[0]])
    origin = pts.mean(axis=0)
    #along, perp, origin = fit_curb_line(pts)

    # signed position along the curb line for each cluster
    along_proj = centered @ along
    order = np.argsort(along_proj)
    sorted_proj = along_proj[order]
    sorted_pts = pts[order]
    sorted_widths = widths[order]
    sorted_heights = heights[order]

    # spacing stats -- spot length along the curb is ~ one car width
    median_spot_length = float(np.median(sorted_widths))
    median_car_height = float(np.median(sorted_heights))

    spots = []

    # 5a. emit the real (clustered) spots
    for i, idx in enumerate(order):
        c = row_clusters[idx]
        spots.append(_make_spot(c, row_id, source=c["source"]))

    # 5b. find gaps and extrapolate
    for i in range(len(sorted_proj) - 1):
        gap = sorted_proj[i + 1] - sorted_proj[i]
        n_extra = int(round(gap / median_spot_length)) - 1
        if n_extra <= 0:
            continue
        needed = n_extra * median_spot_length
        if gap < needed * (1 - gap_tolerance):
            continue

        for k in range(1, n_extra + 1):
            t = sorted_proj[i] + k * (gap / (n_extra + 1))
            extrap_point = origin + t * along
            extrap_cluster = {
                "ground_x": float(extrap_point[0]),
                "ground_y": float(extrap_point[1]),
                "w": median_spot_length,
                "h": median_car_height,
                "n_observations": 0,
                "n_videos": 0,
                "max_dwell_frames": 0,
            }
            spots.append(_make_spot(extrap_cluster, row_id, source="extrapolated"))

    line_info = {
        "origin": origin.tolist(),
        "along_unit": along.tolist(),
        "perp_unit": perp.tolist(),
        "median_spot_length": median_spot_length,
        "median_car_height": median_car_height,
        "n_real_spots": int(len(row_clusters)),
        "n_extrapolated": int(sum(1 for s in spots if s["source"] == "extrapolated")),
    }
    return spots, line_info
    '''