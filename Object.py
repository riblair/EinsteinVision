import numpy as np
import cv2
from Detection import Detection

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
    