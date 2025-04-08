import cv2
import numpy as np
from ultralytics import YOLO
import Utilities as util

from Detection import *
from Object import Object, SpeedSign, TrafficLight

THRESHOLD = 0.25 # traffic lights seem to have confidence from 0.25-0.3

def get_detections_from_image(image: np.ndarray, models: list[YOLO]) -> list[Detection]:
    detections = []
    for model in models:
        results = model(image)
        for result in results:
            for box in result.boxes:
                box = box.to('cpu')
                confidence = box.conf.numpy().flatten()[0]
                class_id = model.names[int(box.cls)]
                print(f"[{class_id}] confidence: {confidence}")
                if confidence < THRESHOLD:
                    continue
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

def refine_objects(image: np.ndarray, objects: list[Object], models: dict):
    refined = []
    for obj in objects:
        print(obj.original_detection.class_id)
        if obj.original_detection.class_id == "warning":
            new_obj = SpeedSign.parse_speed_sign(image, models["OCR"], obj)
            refined.append(new_obj)
        if obj.original_detection.class_id == "traffic light":
            new_obj = TrafficLight.parse_traffic_light(image, obj)
            refined.append(new_obj)
            pass
        else:
            refined.append(obj)
    return refined

def main():
    # NOTE: Temporary code to test functions in this file. Much of this logic can be written in main.py
    cap = cv2.VideoCapture("Videos/scene10_front.mp4") # scene 6 is a disaster...
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    rand_start = 800
    cap.set(cv2.CAP_PROP_POS_FRAMES, rand_start)
    scene_counter = 0

    print("---Loading Model---")
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    model_dict = util.load_models("Models/")

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more framesq
            break
        detections = get_detections_from_image(frame, model_dict["objects"])
        for detection in detections:
            print(f"Found {detection.class_id}")
            cv2.rectangle(frame, np.uint16(detection.top_left), np.uint16(detection.bottom_right), (255, 0, 0), 1)
            cv2.circle(frame, np.uint16(detection.center), 1, (255,255,0), 3)
        print(f"Number of Detections: {len(detections)}")
        
        cv2.imshow("cap", frame)
        cv2.waitKey(5)
        # cv2.destroyAllWindows()
        scene_counter +=1
    pass

if __name__ == "__main__":
    main()
