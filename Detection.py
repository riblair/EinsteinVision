import numpy as np

class Detection:
    def __init__(self, class_id: str, confidence: float, top_left: np.ndarray, bottom_right: np.ndarray):
        self.class_id = class_id.lower()
        self.top_left = top_left
        # Casting to array for consistency with YOLO outputs
        self.width, self.height = bottom_right - top_left
        self.center = np.array([top_left[0]+(self.width/2), top_left[1]+(self.height/2)])
        self.bottom_right = bottom_right
        self.confidence = confidence