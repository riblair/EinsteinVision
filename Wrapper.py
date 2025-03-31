import argparse
import cv2
import json
import math
import numpy as np
import os
import random
import torch

import Line_Detection as ld
import BlenderStuff as bs

def env_setup():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Scene", default="scene1_front.mp4", type=str, help="Path to video file. Default: 'scene1_front.mp4'")
    Parser.add_argument("--Json_Name", default="scenes.json", type=str, help="filename of the json object file. Default:'scenes.json'")
    Parser.add_argument("--Outputs", default="Output/", type=str, help="Path for rendered files. Default:'outputs/'")
    Args = Parser.parse_args()

    os.makedirs(Args.Outputs, exist_ok=True)
    return Args

def main():
    args = env_setup()
    cap = cv2.VideoCapture(args.Scene) # scene 6 is a disaster...
    print("---Loading Model---")
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(device)

    rand_start = random.randint(250,700)
    scene_counter = 0

    data_dictionary = {
            "camera_pose" : [0, 0, 1.2, 1.54, 0.0, 0.0],
            "Scenes" : []
        }
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    print("---Processing Video---")
    while True:
        scene_counter+=1
        object_list = []
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more frames
            break

        if scene_counter < rand_start: continue
        # Can you parse more than one frame at a time through the model? 
        depth_image = zoe.infer_pil(frame) # very slow :/
        print("---Detecting Lane_Lines---")
        lane_line_list = ld.get_line_objects(frame, depth_image)
        object_list.extend(lane_line_list)

        raw_detections = []
        
        # detections will have a center point, that needs to be parsed into a world point

        objects_dict = {
            "scene_num": scene_counter,
            "objects" : [obj.to_json() for obj in object_list]
        }
        data_dictionary["Scenes"].append(objects_dict)
        if scene_counter == rand_start+5:
            break

    print("---Writing to Json---")
    with open(args.Json_Name, 'w') as f:
        f.write(json.dumps(data_dictionary, indent=4))

    print("---Rendering images---")
    bs.render_images(args.Json_Name, args.Outputs)

if __name__ == '__main__':
    main()