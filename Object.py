import numpy as np

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
    
    