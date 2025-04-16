import cv2
import numpy as np

class OpticalFlow():
    def __init__(self, init_frame):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
        self.prev_frame = init_frame
        self.prev_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.prev_pts, self.prev_descriptors = self.sift.detectAndCompute(self.prev_gray, None)
        self.mask = None
        self.flow = None
    
    def update(self, curr_frame):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_pts, curr_descriptors = self.sift.detectAndCompute(curr_gray, None) 
        height, width = curr_gray.shape 
        
        matches = self.matcher.match(self.prev_descriptors, curr_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        
        prev_pts_np = np.float32([self.prev_pts[m.queryIdx].pt for m in matches]).reshape(-1, 2)  
        curr_pts_np = np.float32([curr_pts[m.trainIdx].pt for m in matches]).reshape(-1, 2) 
        
        H, _ = cv2.findHomography(prev_pts_np, curr_pts_np, method=cv2.RANSAC)
        
        warped_frame = cv2.warpPerspective(self.prev_frame, H, (width, height))
        warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        
        self.flow = cv2.calcOpticalFlowFarneback(
                                                    prev=warped_frame,
                                                    next=curr_gray,
                                                    flow=None,
                                                    pyr_scale=0.3,       # Focus on finer details
                                                    levels=3,            # Default pyramid levels
                                                    winsize=9,           # Smaller window for parked car detection
                                                    iterations=10,        # More iterations for better convergence
                                                    poly_n=5,            # Smaller neighborhood for small motion
                                                    poly_sigma=1.0,      # Default smoothing
                                                    flags=0              # No special flags
                                                )
        magnitude, direction = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        parked_mask = cv2.inRange(magnitude, 0, 0.1)  # TUNE ME
        # parked_mask = cv2.cvtColor(parked_mask, cv2.COLOR_BGR2GRAY)
        self.mask = parked_mask
        
        self.prev_gray = curr_gray
        self.prev_pts = curr_pts
        self.prev_frame = curr_frame
        self.prev_descriptors = curr_descriptors
        
        return parked_mask


# def visualize_flow(flow, name="Flow"):
#     # Convert flow to color representation
#     hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = direction
#     hsv[..., 1] = 255  # Full saturation
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Brightness = magnitude
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def main():
    
    cap = cv2.VideoCapture("Videos/scene5_right.mp4")
    ret, prev_frame = cap.read()
    of = OpticalFlow(prev_frame)
    
    starting_frame = 70
    frame_count = 1
    
    while cap.isOpened():
        if frame_count < starting_frame:
            ret, curr_frame = cap.read()
            frame_count += 1
            continue
        ret, curr_frame = cap.read()
        parked_mask = of.update(curr_frame)
        parked_viz = cv2.cvtColor(parked_mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([curr_frame, parked_viz])
        cv2.imshow("Analysis: Keypoints | Optical Flow | Residual Flow | Parked Cars", combined)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
