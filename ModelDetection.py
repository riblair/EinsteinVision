from ultralytics import YOLO
import cv2
import numpy as np
import random
import torch

from Detection import *

def get_detections_from_image(image: np.ndarray, models: list[YOLO]) -> list[Detection]:
    detections = []
    for model in models:
        results = model(image)
        
        for result in results:
            for box in result.boxes:
                class_id = model.names[int(box.cls)]
                u1, v1, u2, v2 = box.xyxy.numpy().flatten()
                confidence = box.conf.numpy().flatten()[0]
                detection = Detection(class_id, confidence, np.array([u1, v1]), np.array([u2, v2]))
                detections.append(detection)
    return detections

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