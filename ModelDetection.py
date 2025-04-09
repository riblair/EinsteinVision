import cv2
import numpy as np
from ultralytics import YOLO
import copy

import Utilities as util
from Detection import *
from Object import Object, SpeedSign, TrafficLight, Person, Car


THRESHOLD = 0.5

def is_person_facing_camera(keypoints: np.ndarray):
    return np.any(keypoints[0:3, 0]) != 0
            
# def visualize_car_segment(image, model: YOLO):
#     results = model(image)
#     for result in results:
#         for mask in results.masks:
            
def get_detections_from_image(image: np.ndarray, models: list[YOLO]) -> list[Detection]:
    detections = []
    for model in models:
        results = model(image)
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
            else:
                keypoints = None
            for box in result.boxes:
                box = box.to('cpu')
                confidence = box.conf.numpy().flatten()[0]
                if confidence < THRESHOLD:
                    continue
                class_id = model.names[int(box.cls)]
                print(f"[{class_id}] confidence: {confidence}")
                if confidence < THRESHOLD:
                    continue
                u1, v1, u2, v2 = box.xyxy.numpy().flatten()
                detection = Detection(class_id, confidence, np.array([u1, v1]), np.array([u2, v2]), keypoints=keypoints)
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

def visualize_detections(raw_detections: list[Detection], image):
    for detection in raw_detections:
        cv2.rectangle(image, np.uint16(detection.top_left), np.uint16(detection.bottom_right), (255, 0, 0), 1)
        cv2.putText(image, detection.class_id, np.uint16(detection.top_left)+10, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)
        cv2.putText(image, str(detection.confidence), np.uint16(detection.top_left)+20, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)
    cv2.imshow("", image)
    if cv2.waitKey(30) == ord('q'):
        return

def refine_objects(image: np.ndarray, objects: list[Object], models: dict):
    refined = []
    for obj in objects:
        # print(obj.original_detection.class_id)
        if obj.original_detection.class_id == "warning":
            # new_obj = SpeedSign.parse_speed_sign(image, models["OCR"], obj)
            # refined.append(new_obj)
            pass
        elif obj.original_detection.class_id == "traffic light":
            new_obj = TrafficLight.parse_traffic_light(image, obj)
            refined.append(new_obj)
            pass
        elif obj.original_detection.class_id == 'person':
            new_obj = Person.parse_person(image, obj, models["human_pose"])
            refined.append(new_obj)
        elif obj.original_detection.class_id == 'car' or obj.original_detection.class_id == 'truck':
            new_obj = Car.parse_car(image, obj, models["car_orient"])
            refined.append(new_obj)
        else:
            refined.append(obj)
    return refined

def main():
    # NOTE: Temporary code to test functions in this file. Much of this logic can be written in main.py
    cap = cv2.VideoCapture("Videos/scene10_front.mp4") # scene 6 is a disaster...
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    rand_start = 720
    counter = 0
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
        if not ret: # no more frames
            break
        # choosing a random frame in the video to begin analysis
        if counter < rand_start:
            counter+=1
            continue
        
        # results = car_segmentation_model(frame)
        # for result in results:
        #     if result.masks is None:
        #         continue
        #     for box in result.boxes:
        #         box = box.to('cpu')
        #         confidence = box.conf.numpy().flatten()[0]
        #         if confidence < THRESHOLD:
        #             continue
        #         class_id = car_segmentation_model.names[int(box.cls)]
        #         print(class_id, confidence)
        #         u1, v1, u2, v2 = box.xyxy.numpy().flatten()
        #     for mask in result.masks:
        #         points = np.int32(mask.xy)
        #         cv2.polylines(frame, points, True, (255, 0, 0), 1)
        
        
        detections = get_detections_from_image(frame, model_dict["objects"])
        # detections = get_person_from_image(frame, human_pose_model)
        anno_frame = copy.deepcopy(frame)
        for detection in detections:
            print(f"Found {detection.class_id}")
            cv2.rectangle(anno_frame, np.uint16(detection.top_left), np.uint16(detection.bottom_right), (255, 0, 0), 1)
            cv2.circle(anno_frame, np.uint16(detection.center), 1, (255,255,0), 3)
            cv2.putText(anno_frame, detection.class_id, np.uint16(detection.center), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
            if detection.class_id == 'person':
                patch = frame[int(detection.top_left[1]):int(detection.bottom_right[1]), int(detection.top_left[0]):int(detection.bottom_right[0])]
                result = human_pose_model(patch)[0]
                keypoints = result.keypoints.data.cpu().numpy()[0, :, :]
                keypoints[:, 0] += detection.top_left[0]
                keypoints[:, 1] += detection.top_left[1]
                detection.keypoints = keypoints
                detection.facing_away = is_person_facing_camera(keypoints)    
                cv2.putText(anno_frame, str(detection.facing_away), np.uint16(detection.center+10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)  
                
            if detection.keypoints is None:
                continue
            for keypoint in detection.keypoints:
                x, y, conf = keypoint
                cv2.circle(anno_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                    
        # print(f"Number of Detections: {len(detections)}")
        
        cv2.imshow("cap", anno_frame)
        if cv2.waitKey(1) == ord('q'):
            return
    pass

if __name__ == "__main__":
    main()
