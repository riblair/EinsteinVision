from ultralytics import YOLO
import numpy as np
import cv2

def main():
    model = YOLO("yolov8n-pose.pt")
    image = cv2.imread('walking3.jpg')
    results = model(image)
    visualized_image = image.copy()
    
    # Process each detection
    for result in results:
        # Extract keypoints if available
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            
            # Process each detected person
            for kpts in keypoints:
                # Draw each keypoint
                for i, (x, y, conf) in enumerate(kpts):
                    if conf > 0.5:  # Only draw keypoints with confidence > 0.5
                        # Draw a circle at each keypoint
                        cv2.circle(visualized_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        
                        # Optionally, add keypoint index
                        cv2.putText(visualized_image, str(i), (int(x), int(y)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw connections between keypoints (skeleton)
                # YOLOv8 keypoint pairs for skeleton
                skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], 
                            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], 
                            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
                
                for p1, p2 in skeleton:
                    # Check if both keypoints are confident
                    if kpts[p1-1][2] > 0.5 and kpts[p2-1][2] > 0.5:
                        # Draw line connecting the keypoints
                        pt1 = (int(kpts[p1-1][0]), int(kpts[p1-1][1]))
                        pt2 = (int(kpts[p2-1][0]), int(kpts[p2-1][1]))
                        cv2.line(visualized_image, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow('Pose Detection', visualized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the image
    # cv2.imwrite('walking_with_keypoints.jpg', visualized_image)
    

if __name__ =='__main__':
    main()