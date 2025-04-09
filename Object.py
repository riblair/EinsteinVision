import numpy as np
import cv2
from Detection import Detection
import Utilities as util

class Object:
    def __init__(self, original_detection: Detection, pose: np.ndarray):
        self.original_detection = original_detection
        self.pose = pose
    
    def to_json(self):
        return {
            "type": self.original_detection.class_id,
            "pose": self.pose.flatten().tolist()
        }
    

class SpeedSign(Object):
    def __init__(self, original_detection, pose, speed_limit:str):
        super().__init__(original_detection, pose)
        self.speed_limit = speed_limit
        
    def to_json(self):
        obj_dict = super().to_json()
        obj_dict["speed_limit"] = self.speed_limit
        return obj_dict
    
    @classmethod
    def parse_speed_sign(cls, image: np.ndarray, ocr_model, obj:Object):
        cropped_detection = obj.original_detection.get_crop(image)
        results = ocr_model.readtext(cropped_detection)
        for result in results:
            # tuple with: (bounding box, text, confidence)
            if not result[1].isdigit(): continue # only add objects 
            # assuming that only 1 detection will be a number.
            print(f"Speed sign says {result[1]}")
            return SpeedSign(obj.original_detection, obj.pose, result[1])
        
class TrafficLight(Object):
    def __init__(self, original_detection, pose, color):
        super().__init__(original_detection, pose)
        self.color = color
    
    def to_json(self):
        obj_dict = super().to_json()
        obj_dict["color"] = self.color
        return obj_dict
    
    @classmethod
    def parse_traffic_light(cls, image: np.ndarray, obj: Object):
        cropped_detection = obj.original_detection.get_crop(image)
        thirds = int(cropped_detection.shape[0]/3)
        red_area = cv2.cvtColor(cropped_detection[0:thirds, :], cv2.COLOR_BGR2LAB)
        yellow_area = cv2.cvtColor(cropped_detection[thirds:2*thirds, :], cv2.COLOR_BGR2LAB)
        green_area = cv2.cvtColor(cropped_detection[2*thirds:, :], cv2.COLOR_BGR2LAB)

        red_lum = np.mean(red_area[:,:,0])
        yellow_lum = np.mean(yellow_area[:,:,0])
        green_lum = np.mean(green_area[:,:,0])

        dom_color = ""

        if(red_lum > green_lum):
            dom_color = "Red" if (red_lum > yellow_lum) else "Yellow"
        else:
            dom_color = "Green" if (green_lum > yellow_lum) else "Yellow"
        print(f"dominiant color is {dom_color}")
        return TrafficLight(obj.original_detection, obj.pose, dom_color)
    
class Person(Object):
    def __init__(self, original_detection: Detection, pose: np.ndarray):
        super().__init__(original_detection, pose)
        self.left_elbow = 0.0
        self.right_elbow = 0.0
        self.left_shoulder = 0.0
        self.right_shoulder = 0.0
        self.left_hip = 0.0
        self.right_hip = 0.0
        self.left_knee = 0.0
        self.right_knee = 0.0
        
    def to_json(self):
        obj_dict = super().to_json()
        obj_dict["LeftShoulder"] = self.left_shoulder
        obj_dict["RightShoulder"] = self.right_shoulder
        obj_dict["LeftElbow"] = self.left_elbow
        obj_dict["RightElbow"] = self.right_elbow
        obj_dict["LeftHip"] = self.left_hip
        obj_dict["RightHip"] = self.right_hip
        obj_dict["LeftKnee"] = self.left_knee
        obj_dict["RightKnee"] = self.right_knee
        return obj_dict
    
    def extract_keypoints(self, image, pose_model):
        # Extracts the human pose keypoints and saves them to original_detection.keypoints
        patch = self.original_detection.get_crop(image)
        result = pose_model(patch)[0]
        self.original_detection.keypoints = result.keypoints.data.cpu().numpy()[0, :, :]
        self.original_detection.keypoints[:, 0] += self.original_detection.top_left[0]
        self.original_detection.keypoints[:, 1] += self.original_detection.top_left[1]
    
    def get_orientation_from_detection(self):
        if self.original_detection.keypoints is None or self.original_detection.keypoints.size == 0:
            return
        angle = 0  # 0 rad is facing the camera
        if np.all(self.original_detection.keypoints[0:3, 0]) == 0:  # We cannot see any facial features, they are facing away from the camera
            angle = np.pi
        self.pose[5, 0] = angle
        
    def get_joint_angles_from_keypoints(self):
        if self.original_detection.keypoints is None or self.original_detection.keypoints.size == 0:
            return
        left_shoulder = self.original_detection.keypoints[5, :]
        right_shoulder = self.original_detection.keypoints[6, :]
        left_elbow = self.original_detection.keypoints[7, :]
        right_elbow = self.original_detection.keypoints[8, :]
        left_wrist = self.original_detection.keypoints[9, :]
        right_wrist = self.original_detection.keypoints[10, :]
        
        self.left_shoulder = util.three_point_angle_2D(right_shoulder, left_shoulder, left_elbow)
        self.right_shoulder = -util.three_point_angle_2D(left_shoulder, right_shoulder, right_elbow)
        self.left_elbow = -util.three_point_angle_2D(left_shoulder, left_elbow, left_wrist)
        self.right_elbow = -util.three_point_angle_2D(right_shoulder, right_elbow, right_wrist)
        
        left_hip = self.original_detection.keypoints[11, :]
        right_hip = self.original_detection.keypoints[12, :]
        left_knee = self.original_detection.keypoints[13, :]
        right_knee = self.original_detection.keypoints[14, :]
        left_ankle = self.original_detection.keypoints[15, :]
        right_ankle = self.original_detection.keypoints[16, :]
        
        self.left_hip = util.three_point_angle_2D(right_shoulder, left_hip, left_knee)
        self.right_hip = -util.three_point_angle_2D(left_shoulder, right_hip, right_knee)
        self.left_knee = -util.three_point_angle_2D(left_hip, left_knee, left_ankle)
        self.right_knee = -util.three_point_angle_2D(right_hip, right_knee, right_ankle)
        
        print(f"""
              Left Shoulder: {self.left_shoulder} \n
              Right Shoulder: {self.right_shoulder} \n
              Left Elbow: {self.left_elbow} \n
              Right Elbow: {self.right_elbow} \n
              Left Hip: {self.left_hip} \n
              Right Hip: {self.right_hip} \n
              Left Knee: {self.left_knee} \n
              Right Knee: {self.right_knee} \n
              """)
        
    @classmethod
    def parse_person(cls, image: np.ndarray, obj: Object, human_pose_model):
        new_obj = Person(obj.original_detection, obj.pose)
        new_obj.extract_keypoints(image, human_pose_model)
        new_obj.get_orientation_from_detection()
        new_obj.get_joint_angles_from_keypoints()
        return new_obj

class Car(Object):
    def __init__(self, original_detection, pose, direction):
        super().__init__(original_detection, pose)
        self.direction = "back"
    
    @classmethod
    def parse_car(cls, image, obj: Object, orientation_model):
        patch = obj.original_detection.get_crop(image)
        result = orientation_model(patch)[0]
        for box in result.boxes:
            box = box.to('cpu')
            confidence = box.conf.numpy().flatten()[0]
            side = orientation_model.names[int(box.cls)]
        return Car(obj.original_detection, obj.pose, side)
            
            
            