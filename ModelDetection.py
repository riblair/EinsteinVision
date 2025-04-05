from ultralytics import YOLO
import cv2
import numpy as np
import random
import torch
import Utilities as util

from Detection import *
from Object import Object


THRESHOLD = 0.6

def get_detections_from_image(image: np.ndarray, models: list[YOLO]) -> list[Detection]:
    detections = []
    for model in models:
        results = model(image)
        for result in results:
            for box in result.boxes:
                confidence = box.conf.numpy().flatten()[0]
                # if confidence < THRESHOLD:
                #     continue
                class_id = model.names[int(box.cls)]
                u1, v1, u2, v2 = box.xyxy.numpy().flatten()
                detection = Detection(class_id, confidence, np.array([u1, v1]), np.array([u2, v2]))
                detections.append(detection)
    return detections

def detections_to_world(raw_detections: list[Detection], depth_image: np.ndarray) -> list[Object]:
    localized_objects = []
    for detection in raw_detections:
        center_pixel = detection.center
        depth = depth_image[int(center_pixel[1]),int(center_pixel[0])]
        position = util.pixel_to_world(center_pixel, float(depth))
        zero_mat = np.array([[0], [0], [0]])
        pose = np.vstack((position, zero_mat))
        localized_objects.append(Object(detection, pose))
    return localized_objects

def main():
    # NOTE: Temporary code to test functions in this file. Much of this logic can be written in main.py
    cap = cv2.VideoCapture("Videos/scene10_front.mp4") # scene 6 is a disaster...
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sign_model = YOLO("best.pt", verbose=False)  # General YOLO Model for Vehicles, Traffic Lights, and Pedestrians
    general_model = YOLO("yolov8n.pt", verbose=False)
    rand_start = random.randint(700,1000)
    counter = 0
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more framesq
            break
        # choosing a random frame in the video to begin analysis
        if counter < rand_start:
            counter+=1
            continue
        detections = get_detections_from_image(frame, [general_model, sign_model])
        for detection in detections:
            print(f"Found {detection.class_id}")
            cv2.rectangle(frame, np.uint16(detection.top_left), np.uint16(detection.bottom_right), (255, 0, 0), 1)
            cv2.circle(frame, np.uint16(detection.center), 1, (255,255,0), 3)
        print(f"Number of Detections: {len(detections)}")
        
        cv2.imshow("cap", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    pass

if __name__ == "__main__":
    main()