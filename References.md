### Project 3 References
For object detection (vehicles, pedestrians, road signs, traffic signals), we implemented two versions of the YOLO model available by Ultralytics.

https://docs.ultralytics.com/models/yolov8/#performance-metrics
- The first model used YOLOv8n, a detection model trained using the COCO dataset. This model has 80 pre-trained classes. We use this model to detect vehicles, pedestrians, and traffic lights.

https://universe.roboflow.com/us-traffic-signs-pwkzx/us-road-signs 
- The second YOLO model wwas trained on a public dataset available on RoboFlow. The dataset targets non-pavement traffic signs i.e. stop signs, one way, speed limit, etc. We trained a YOLO model using this dataset on WPI's turing cluster (500 epochs, 300 iterations per epoch).


