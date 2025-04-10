import argparse
import cv2
import json
import math
import numpy as np
import os
import random
import torch

from ultralytics import YOLO

import Line_Detection as ld
import BlenderStuff as bs
import Utilities as util
import ModelDetection as md
from Detection import Detection
from Object import Object

def env_setup():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Scene", default="Videos/scene10_front.mp4", type=str, help="Path to video file. Default: 'scene1_front.mp4'")
    Parser.add_argument("--Start", default="-1", type=int, help="Frame to start processing on")
    Parser.add_argument("--Json_Name", default="Scenes/scenes.json", type=str, help="filename of the json object file. Default:'scenes.json'")
    Parser.add_argument("--Outputs", default="Output/", type=str, help="Path for rendered files. Default:'outputs/'")
    Parser.add_argument("--Video_Name", default="output.mp4", type=str, help="Name of Output video.")
    Args = Parser.parse_args()

    os.makedirs(Args.Outputs, exist_ok=True)
    os.makedirs(Args.Outputs+"Video/", exist_ok=True)
    return Args

# TODO list:
# URGENT: Correct Projection equations
# URGENT: Try different Depth Models 
# Med Priority: Color on Traffic Signs
# Med Priority: Pedestrain Pose
# Med Priority: Traffic Sign Graphics
# Low Priority: Objects (trafic cones, trashcans)
# Low Priority: Improve Line Detection pipeline - See Line_Detection.py

def main():
    args = env_setup()

    cap = cv2.VideoCapture(args.Scene)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    if args.Start == -1:
        rand_start = 860
    else:
        rand_start = args.Start
    cap.set(cv2.CAP_PROP_POS_FRAMES, rand_start)
    scene_counter = -1

    print("---Loading Model---")
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    model_dict = util.load_models("Models/")

    data_dictionary = {
            "camera_pose" : [0, 0, 1.9, 1.54, 0.0, 0.0],
            "Scenes" : []
        }
    print("---Processing Video---")
    while True:
        scene_counter+=1
        if not scene_counter % 6 == 0:
            continue
        object_list = []
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more frames
            break
        # Can you parse more than one frame at a time through the model? 
        depth_image = model_dict["depth"].infer_pil(frame) # very slow :/
        print("---Detecting objects in scene---")
        # Run detection models on image to get Detection objects
        raw_detections = md.get_detections_from_image(frame, model_dict["objects"])
        # md.visualize_detections(raw_detections, frame)
        print("---Detecting Lane_Lines---")
        lane_line_list = ld.get_line_objects(frame, depth_image, raw_detections)
        object_list.extend(lane_line_list)

        localized_objects = md.detections_to_world(raw_detections, depth_image)
        refined_objects = md.refine_objects(frame, localized_objects, model_dict)

        object_list.extend(refined_objects)

        objects_dict = {
            "scene_num": scene_counter+rand_start,
            "objects" : [obj.to_json() for obj in object_list]
        }
        data_dictionary["Scenes"].append(objects_dict)
        
        if scene_counter >= 180:
            break
        
    print("---Writing to Json---")
    with open(args.Json_Name, 'w') as f:
        f.write(json.dumps(data_dictionary, indent=4))

    print("---Rendering images---")
    bs.render_images(args.Json_Name, args.Outputs, args.Scene, rand_start)
    bs.directory_to_video(args.Outputs+"Video/", args.Video_Name)

if __name__ == '__main__':
    main()
