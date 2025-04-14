import copy
import cv2
import easyocr
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

# graphing constants
COLORS = ['Red','Green', 'Blue', 'Orange', 'Black']

# Line Detection Constants 
LOW_WHITE, HIGH_WHITE = 150, 255
LOW_H, HIGH_H = 10, 80
LOW_S, HIGH_S = 49,255
LOW_V, HIGH_V = 112,255
LOW_SIG, HIGH_SIG = 30,80

MAX_CLUSTERS = 5

# Camera Constants
K_MAT = np.array([[1594.7,         0,    655.3],
                  [     0,    1607.7,    414.4],
                  [     0,         0,        1]])

WORLD_TO_CAM = np.array([[1,  0,  0],
                         [0,  0,  -1],
                         [0, 1,  0]])

x_rad = -0.3
X_ROT = np.array([[1, 0, 0],
                  [0, math.cos(x_rad), -math.sin(x_rad)],
                  [0, math.sin(x_rad), math.cos(x_rad)]])

WORLD_TO_CAM_R = WORLD_TO_CAM @ X_ROT
WORLD_TO_CAM_T = np.array([[0], [-1.9], [0]])

MAN_ADJUSTMENTS = np.array([])

HIGH_THETA = 1.8
LOW_THETA = 1.4

# RANSAC parameters
MAX_ITER = 150
LOSS_THRESH = 0.035
PERCENT_CUTOFF = 80

def load_models(model_path) -> dict:
    models = dict([("objects", [])])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models["depth"] = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(device)
    models["objects"].append(YOLO(f"{model_path}general.pt", verbose=False).to(device))
    models["objects"].append(YOLO(f"{model_path}traffic_signs1.pt", verbose=False).to(device))
    models["OCR"] = easyocr.Reader(['en'], model_storage_directory="~/.EasyOCR/model", user_network_directory="~/.EasyOCR/user_network")
    models["car_orient"] = YOLO(f"{model_path}classification.pt", verbose=False).to(device)
    models["human_pose"] = YOLO(f"{model_path}yolo11n-pose.pt", verbose=False).to(device)
    return models

def pixel_arr_projection(pixel_arr: np.ndarray, depth_pixels: np.ndarray) -> np.ndarray:
    """ Projects an array of pixels into R3 using the static front camera transform
        Args:
            pixel_arr (np.ndarray): array of pixels, in shape SX3 [[u,v,1], [u,v,1], ...]
            depth_pixels (np.ndarray): array of depths for each pixel 
        Returns:
            world_points (np.ndarray): array of world coordinates w.r.t blender coordinate frame
    """
    camera_points = (np.linalg.inv(K_MAT) @ pixel_arr.T).T * depth_pixels[:, np.newaxis]
    # homogenous_camera_points = np.hstack((camera_points, np.ones((camera_points.shape[0], 1))))
    world_points = (WORLD_TO_CAM_R.T @ camera_points.T).T - (WORLD_TO_CAM_R.T @ WORLD_TO_CAM_T).T
    return world_points[:, :3] 

def pixel_to_world(pixel: np.ndarray, depth: float):
    homogenous_pixel = np.hstack((pixel, 1)).reshape((3,1)) # 3x1 
    camera_point = (np.linalg.inv(K_MAT) @ homogenous_pixel) * depth # 
    world_point = WORLD_TO_CAM_R.T @ camera_point - WORLD_TO_CAM_R @ WORLD_TO_CAM_T
    world_point[0] *=10
    world_point[1] *=10
    world_point[2] -=2

    return world_point

def add_lines(frame, lines, color):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
            pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
            if theta < HIGH_THETA and theta > LOW_THETA: # lines must be greater than 103 deg OR less than 80 deg 
                cv2.line(frame, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
            else:
                cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
    return frame

def draw_lane_lines(frame, lines):
    for i in range(lines.shape[0]):
        rho = lines[i][0]
        theta = lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        cv2.line(frame, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
    return frame

def graph_raw_lines(lines):
    rhos = []
    thetas = []

    if lines is not None:
        for line in lines:
            rhos.append(float(line[0][0]))
            thetas.append(float(line[0][1]))
        
    plt.scatter(rhos, thetas)
    plt.show()

def graph_clusters(kmeans, lines):
    if kmeans is None:
        graph_raw_lines(lines)
        return
    cluster_num = len(kmeans.cluster_centers_)

    for i in range(cluster_num):
        rhos = []
        thetas = []
        for j in range(len(lines)):
            label = kmeans.labels_[j]
            if label == i:
                rhos.append(float(lines[j][0][0]))
                thetas.append(float(lines[j][0][1]))
        plt.scatter(rhos, thetas, c=COLORS[i])
    plt.show()

def show_images(edges, white_pixels, white_edges, yellow_edges, frame, lines_W):
    scale = 0.33
    time_scale = 27
    cv2.imshow("edge_map", cv2.resize(edges, None, fx=scale, fy=scale))
    cv2.imshow("white_pixels", cv2.resize(white_pixels, None, fx=scale, fy=scale))
    # cv2.imshow("yellow_pixels", cv2.resize(yellow_pixels, None, fx=scale, fy=scale))
    cv2.imshow("White Edges", white_edges)
    cv2.imshow("Yellow Edges", yellow_edges)
    cv2.imshow("original", frame)
    # cv2.imshow("GT", cv2.resize(frame, None, fx=scale, fy=scale))
    # if counter > 500 and counter % 50 == 0:
    #     graph_raw_lines(lines_W)
    cv2.waitKey(time_scale)

def show_line_pixels(line_pixels):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    for i in range(len(line_pixels)):
        ax.scatter(line_pixels[i][:,1], line_pixels[i][:,0], c=COLORS[i])

def show_line_points(line_points, ax_ref=None):
    if ax_ref is None:
        ax = plt.figure().add_subplot(projection='3d') if ax_ref == None else ax_ref
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')
        # ax.set_ylim((0,10))
        # ax.set_xlim((-5,5))
        # ax.set_zlim((-1, 5))
    else:
        ax = ax_ref
    for i in range(len(line_points)):
        ax.scatter(line_points[i][:,0],line_points[i][:,1],line_points[i][:,2], c=COLORS[i])
    # plt.show()

def show_depth_image(depth_image):
    def map_to_range(arr, min_val, max_val, new_min, new_max):
            return [new_min + (x - min_val) * (new_max - new_min) / (max_val - min_val) for x in arr]
    remapped = map_to_range(depth_image, np.min(depth_image), np.max(depth_image), 0, 255)
    remapped = np.array(remapped, dtype=np.uint8)
    cv2.imshow('depth', remapped)

def visualize_stuff(frame, lines, depth_image):
    
    frame = draw_lane_lines(frame, lines)
    show_depth_image(depth_image)
    cv2.imshow('frame', frame)
    
    cv2.waitKey(100)
    plt.show()

def show_direction_RANSAC(best_direction_list, best_inliers_list, line_origins = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    show_line_points(best_inliers_list, ax)
    print(line_origins)
    for i in range(len(best_direction_list)):
        p0 = line_origins[i] if line_origins is not None else best_inliers_list[i][0]
        px = p0 + 5* best_direction_list[i]
        ax.plot([float(p0[0]), float(px[0])], [float(p0[1]), float(px[1])], [float(p0[2]), float(px[2])], color=COLORS[i])
        
def three_point_angle_2D(point1, point2, point3):
    vec1 = point1-point2
    vec2 = point2-point3
    return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)))

x_offset = 90
y_offset = 70
vert_offset = 20

tl_l = (0,              y_offset-vert_offset)
tr_l = (x_offset,       y_offset-vert_offset)
bl_l = (0,          255-y_offset-vert_offset)
br_l = (x_offset,   255-y_offset-vert_offset)

tl_r = (255-x_offset,       y_offset-vert_offset)
tr_r = (255,                y_offset-vert_offset)
bl_r = (255-x_offset,   255-y_offset-vert_offset)
br_r = (255,            255-y_offset-vert_offset)

HEAD_LIGHT_THRESH = 0.8

def thresh_headlights(image: np.ndarray) -> list:
    image = cv2.resize(image, (256, 256))
    left_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(left_mask, tl_l, br_l, 255, -1)

    right_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(right_mask, tl_r, br_r, 255, -1)
    
    frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (0, 0, 196), (26, 61, 255))
    frame_threshold = cv2.erode(frame_threshold, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    frame_threshold = cv2.dilate(frame_threshold, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    frame_threshold_left = cv2.bitwise_and(copy.deepcopy(frame_threshold), left_mask)
    frame_threshold_right = cv2.bitwise_and(copy.deepcopy(frame_threshold), right_mask)

    # print(f"{im_name} Pixels that are lit {np.sum(frame_threshold / 255) * 100 / (90*115)}%")
    # cv2.imshow(f"original resized {im_name}", image)
    # cv2.imshow(f"threshed {im_name}", frame_threshold)
    # cv2.waitKey()

    result_tup = [0, 0]
    if np.sum(frame_threshold_left / 255) * 100 / (90*115) > HEAD_LIGHT_THRESH: result_tup[0] = 1
    if np.sum(frame_threshold_right / 255) * 100 / (90*115) > HEAD_LIGHT_THRESH: result_tup[1] = 1
    return result_tup
