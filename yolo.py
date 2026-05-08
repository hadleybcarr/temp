import cv2
from pathlib import Path
from ultralytics import YOLO
import os
from sklearn.cluster import DBSCAN
import numpy as np
import json 
import pickle

'''All videos in processed folders:
    - cars_adjusting
    - driving_to_parked
    - cars_leaving
    - multicar_movement
    - parked_to_parking
    - white_car_pullup
    - two_cars_leave
    - parallel_park_pullup
    - cars_leave_cars_park
    - one_park_one_leave
    - one_car_pullout
    - gray_car_leaving
    - cars_staying_and_leaving
    - several_car_movement
    - multicar
    - outside_of_bounds
    - parallel_park
    - car_leaving_thayer
    - white_parallel_parking
    - gray_pullout
    - two_car_thayer_parking
    - white_parallel_thayer
    - black_car_leaving_thayer
    - gray_car_pullin_thayer
    - thayer_turnover
    - parallel_parking_process
    - thayer_dark_gray_car
    - two_cars_leaving   
'''       

'''
World visualizations:
- pickup_angle_1-5
- thayer_angle 1-4, 6
'''

'''
Yolo 11
- thayer_angle 5
'''
models = ["yolov8s-worldv2.pt", "yolo11m.pt", "yolo8m.pt", "yolo8n.pt", "visDrone.pt"]
model = YOLO(models[4])

#List of coco labels: https://gist.github.com/tersekmatija/9d00c4683d52d94cf348acae29e8db1a 
ACCEPTED_CLASSES = {
    2, #car 
    3, #motorbike 
    6, #truck
    7, #boat 
    9, #traffic light
    76, #cellphone
}

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

videos_by_camera_old = {
    "sci_li_video": ["lines_car_leaving.mov", "lines_car_parking.mov"]
}

def video_to_image(video_path):
    print(video_path)
    full_video_path = "videos/"+video_path
    print(full_video_path)
    cap = cv2.VideoCapture(full_video_path)
    stem = Path(video_path).stem
    output_directory = Path("photos") / stem
    if not os.path.exists(output_directory):
        Path(f"photos/{stem}").mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:  break
        cv2.imwrite(str(output_directory/f"frame{i:04d}.jpg"), frame)
        i += 1

def detect_images(file_path):
    i = 0
    stem = Path(file_path).stem
    directory = Path(f"photos/{stem}")
    output_directory = Path("yolo_world") / stem
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            detections = model(file_path)
            detections[0].plot()
            detections[0].save(str(output_directory/f"annotated{i:04d}.jpg"))
            i += 1     

def video_to_yolo_label(file_path, model):
    print("Going through", file_path)
    cap = cv2.VideoCapture("videos/" + file_path)
    frame_idx = 0
    all_results = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, imgsz=1280, conf=0.15, verbose=False)
        for box in results[0].boxes:
            if int(box.cls[0]) not in ACCEPTED_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w,  h  = x2 - x1, y2 - y1
            all_results.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx, "cy": cy,
                "w":  w,  "h":  h,
                "ground_x": cx,                       # bottom-center x
                "ground_y": y2,                       # bottom-center y (wheels)
                "score": float(box.conf[0]),
                "source": "yolo_only",
                "frame_idx": frame_idx,
                "video": file_path,
            })
        del results
        frame_idx += 1

    print("Yolo completed on videos...")
    cap.release()
    return all_results        # plain list -- infer_spots.py iterates with .get()

'''Used for running the initial YOLO model'''
def results_spot_detection(results):
    arr = np.array(results)
    if not results:
        return {}
    print("results are", results)
    centers = arr[:, :2]
    sizes = arr[:, 2:]


    #eps is the clustering distance between pixels, can adjust to predict more/less praking spots accordingly
    labels = DBSCAN(eps=40, min_samples=20).fit_predict(centers)
    spots = {}
    for label in set(labels) - {-1}:
        mask = labels == label
        cx, cy = centers[mask].mean(axis=0)
        w,h = np.median(sizes[mask], axis=0)
        spots[f"spot{label}"] = (float(cx-w/2), float(cy-h/2), float(cx+w/2), float(cy + h/2))

    return spots

def cluster_and_classify(detections):
    centers = np.array([[d["cx"], d["cy"]] for d in detections])
    sizes = np.array([[d["w"], d["h"]] for d in detections])
    frames = np.array([d["frame_idx"] for d in detections])
    labels = DBSCAN(eps=40, min_samples=20).fit_predict(centers)
    spots = {}
    for label in set(labels) - {-1}:
        mask = labels == label
        cx, cy = centers[mask].mean(axis=0)
        w,h = np.median(sizes[mask], axis=0)
        cluster_frames = sorted(frames[mask].tolist())
        max_dwell = max_consecutive_run(cluster_frames)
        spots[f"spot_{label}"] = {
            "box": [float(cx-w/2), float(cy-h/2), float(cx+w/2), float(cy+h/2)],
            "source": "clustered" if max_dwell >= 30 else "rejected_road",
            "n_observations": int(mask.sum()),
            "max_dwell_frames": int(max_dwell),
        }

    spot_centers = np.array([
        [(s["box"][0] + s["box"][2])/2, (s["box"][1] + s["box"][3])/2]
        for s in spots.values()
        if s["source"] == "clustered"], dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(spot_centers, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy])
    projections = ((centers - [x0, y0]) @ direction)
    projections.sort()

    gaps = np.diff(projections)
    typical_gap = np.median(gaps)


    for i, gap in enumerate(gaps):
        n_missing = round(gap / typical_gap ) - 1
        missing_p = 0
        for k in range(n_missing):
            offset = projections[i] + typical_gap * (k+1)
            center_2d = np.array([x0, y0]) + offset * direction
            typical_w, typical_h = np.median([
                [(s["box"][0] + s["box"][2])/2, (s["box"][1] + s["box"][3])/2]
                for s in spots.values() if s["source"] == "clustered"], axis=0)
            cx2, cy2 = center_2d
            spots[f"missing_{missing_p}"] = {
                "box": [float(cx2 - typical_w/2), float(cy2-typical_h/2), float(cx2+typical_w/2), float(cy2+typical_h/2)],
                "source": "missing",
                "n_observations": 0,
                "max_dwell_frames": 0
            }
            missing_p += 1

    return spots


def max_consecutive_run(frame_ids, gap_tolerance=2):
    if not frame_ids: return 0
    longest = current = 1
    for prev, curr in zip(frame_ids, frame_ids[1:]):
        if curr - prev <= gap_tolerance:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


#-----------------Code used to make annotated YOLO images-----------------#
#output_directory = Path(f"photos")
#for folder in output_directory.iterdir():
#    detect_images(file_path=folder)

#video_to_image(video_path=data_file_path)

print("Starting to detect spots...")
CACHE  = Path(f"yolo_visDrone_cache")
CACHE.mkdir(exist_ok=True)

for cam, vids in videos_by_camera.items():
    cache_file = CACHE / f"{cam}.pkl"
    if cache_file.exists():
        print(f"{cam} cached, skipping")
        continue
    print(f"=== {cam} ===")
    detections = []
    for v in vids:
        #if cam == "thayer_angle_5":
        #    model = model_11
        #else:
        #    model = world_model
        detections.extend(video_to_yolo_label(v, model=model))
        with open(cache_file, "wb") as f:
            pickle.dump(detections,f)
    
    
    #spots = results_spot_detection(detections)
    #spots_by_camera[cam] = spots 
    #Path(f"yolo_world_spots_{cam}.json").write_text(
    #json.dumps(spots_by_camera, indent=2))
#spots_by_camera = {}
#for cache_file in Path(CACHE).glob("*.pkl"):
#cache_file = "{model}.pkl"
#cam = cache_file.stem
#detections = pickle.load(open(cache_file, "rb"))
#spots_by_camera[cam] = cluster_and_classify(detections)
