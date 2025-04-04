import copy
import cv2
import math
import numpy as np
import random
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import Utilities as util 
from Detection import Detection
# TODO:
# 1 Lane solution / small dataset detections
# outlier rejection
# Reject data that is too far away from detected lines
# Implement some consistency between frames perhaps 

class Line():
    def __init__(self, x_offset):
        self.x_offset = x_offset

    def to_json(self):
        return {
            "type": "lane_line",
            "pose": [self.x_offset, 0, 0, 0, 0, 0]
        }

def clean_data(lines):
    # TODO - also reject vertical lines (that aren't centered perhaps)
    cleaned_lines = []
    if lines is not None:
        for line in lines:
            if line[0][1] > util.HIGH_THETA or line[0][1] < util.LOW_THETA:
                cleaned_lines.append(line)
    return cleaned_lines

def cluster_lines(lines):
    line_arr = np.array(lines).squeeze(1)
    scores = []
    kmeans_out = []
    if len(line_arr) < 7: # not enough data to be clustering? NOT ELEGANT, FIX!
        return None
    total_variance = np.var(line_arr, axis=0).sum()
    if total_variance < 1:
        return None 

    for k in range(2, util.MAX_CLUSTERS+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(line_arr)
        kmeans_out.append(kmeans)
        score = silhouette_score(line_arr, kmeans.labels_)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    return kmeans_out[best_k-2]

def detect_lines(frame: np.ndarray, detection_mask: np.ndarray) -> np.ndarray: 
    imshape = frame.shape

    """Convert frame to hsv for detecting yellow pixels, and hls for detecting white pixels """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    """ Color Threshholding and edge detection """
    white_pixels = cv2.inRange(frame_hls, (0,util.LOW_WHITE, 0), (255,util.HIGH_WHITE, 255))
    yellow_pixels = cv2.inRange(frame_hsv, (util.LOW_H,util.LOW_S,util.LOW_V), (util.HIGH_H,util.HIGH_S,util.HIGH_V))
    edges = cv2.Canny(frame, util.LOW_SIG, util.HIGH_SIG)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

    white_edges = cv2.bitwise_and(white_pixels, edges)
    yellow_edges = cv2.bitwise_and(yellow_pixels, edges)

    """Remove false matches """
    # truncate entries above the halfway line on the image
    white_edges[0:int(imshape[0]/2), :] = 0
    yellow_edges[0:int(imshape[0]/2), :] = 0

    # create a mask from detected objects. As detected objects obscure points on the line
    white_edges = cv2.bitwise_and(white_edges, detection_mask)
    yellow_edges = cv2.bitwise_and(yellow_edges, detection_mask)

    """Extract lines from image using cv2 HoughLines"""
    lines_W = cv2.HoughLines(white_edges, 1, np.pi / 180, 80, None, 0, 0)
    lines_Y = cv2.HoughLines(yellow_edges, 1, np.pi / 180, 80, None, 0, 0)

    lines_W = clean_data(lines_W)
    lines_Y = clean_data(lines_Y)

    all_lines = lines_W + lines_Y
    
    """Cluster detected lines into groups and extract centroids """
    if len(all_lines):
        kmeans = cluster_lines(all_lines)
        if kmeans:
            return kmeans.cluster_centers_
    return None

def valid_pix(u, v, shape, mask): # reject pixels that are outside of frame and above the half-way mark
    return v > 0 and u > 0 and v < shape[0]/2 and u < shape[1] and mask[v,u] != 0

def find_points_on_line(lines, imshape, detection_mask: np.ndarray):
    pix_space = np.linspace(-1000, 1000, 2001, dtype=np.int16).reshape(2001, 1)
    line_pixels = []

    for i in range(len(lines)):

        rho = lines[i][0]
        theta = lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        u0 = a * rho
        v0 = b * rho

        us = pix_space[:] * -b + u0
        vs = pix_space[:] * a + v0

        pixels = np.hstack((us,vs)).astype(np.int16)
        pixels_in_range = np.array([pixel for pixel in pixels if valid_pix(pixel[0], pixel[1], imshape, detection_mask)])
        
        unique_pixels = np.unique(pixels_in_range, axis=0)
        line_pixels.append(unique_pixels)

    return line_pixels

def pixels_to_world_points(depth_image, line_pixels):

    line_point_list = []
    for i in range(len(line_pixels)): 
        if len(line_pixels[i]) == 0:
            continue
        depths = np.array([depth_image[pixel[1], pixel[0]] for pixel in line_pixels[i]])

        homogenous_pixels = np.hstack((line_pixels[i], np.ones((line_pixels[i].shape[0],1))))
        line_points = util.pixel_arr_projection(homogenous_pixels, depths)
        line_point_list.append(line_points)

    # MxNx3, where M is the number of detected lines. N is defined above.
    return line_point_list

def get_ray_from_points(line_points_list): 
    # TODO: add penalty for directions dominated by Z
    best_direction_list = []
    best_inliers_list = []
    for i in range(len(line_points_list)):
        best_direction = None
        best_inliers = []
        line_points = line_points_list[i]
        for j in range(util.MAX_ITER):
            indices = np.random.choice(line_points.shape[0], size=2)
            direction =  line_points[indices[0]] - line_points[indices[1]]
            unit_d = direction / np.linalg.norm(direction)
            if unit_d[1] < 0: # ensures ray is pointing away from camera
                unit_d = -unit_d
            inliers = []
            for point in line_points:
                loss = np.linalg.norm(np.cross(point - line_points[indices[1]], unit_d))
                if loss < util.LOSS_THRESH: # arbitrarily chosen threshold
                    inliers.append(point)
            if len(inliers) > len(best_inliers):
                percentage = 100 * len(inliers)/len(line_points)
                print(f"percentage of inliers found: {round(percentage,2)}%")
                best_inliers = inliers
                best_direction = unit_d
                if percentage > util.PERCENT_CUTOFF:
                    break
        # COULD RUN-secondary ransac to create even better direction estimate
        best_direction_list.append(best_direction)
        best_inliers_list.append(np.array(best_inliers))
    
    return best_direction_list, best_inliers_list

def get_origins_from_points(best_direction_list, best_inliers_list):
    origins = []
    for i in range(len(best_direction_list)):
        p0 = best_inliers_list[i][-1] 
        t = -p0[1] / best_direction_list[i][1]
        origin = p0 + t* best_direction_list[i]
        origins.append(origin)
    return origins


def get_line_objects(frame: np.ndarray, depth_image: np.ndarray, detections: list[Detection]) -> list[Line]:
    
    obj_bitmask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
    for obj in detections:
        obj_bitmask[int(obj.top_left[1]):int(obj.bottom_right[1]), int(obj.top_left[0]):int(obj.bottom_right[0])] = 0

    lines = detect_lines(frame, obj_bitmask)
    if lines is None:
            return []
    line_pixels = find_points_on_line(lines, frame.shape[:2], obj_bitmask)
    line_points = pixels_to_world_points(depth_image, line_pixels)
    util.show_line_points(line_points)
    best_direction_list, best_inliers_list = get_ray_from_points(line_points)
    line_origins = get_origins_from_points(best_direction_list, best_inliers_list)
    line_objs = []
    for origin in line_origins:
        line_objs.append(Line(origin[0]))
    
    util.show_direction_RANSAC(best_direction_list, best_inliers_list, line_origins)
    util.visualize_stuff(frame, lines, depth_image)
    return line_objs

def main():
    cap = cv2.VideoCapture("Videos/scene1_front.mp4") # scene 6 is a disaster...
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(device)

    rand_start = random.randint(250,700)
    print(rand_start)
    counter = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more frames
            break
        # choosing a random frame in the video to begin analysis
        if counter < rand_start:
            counter+=1
            continue

        get_line_objects(frame, zoe.infer_pil(frame))

if __name__ == '__main__':
    main()