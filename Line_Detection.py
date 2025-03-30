import cv2
import math
import numpy as np
import random
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import Utilities as util 
# Key improvements that need to be made
# Reject Thetas in another range (ones that are too vertical)
# Implement 1 lane solution
# Reject data that is too far away from detected lines
# Implement some consistency between frames perhaps 

def clean_data(lines):
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

def detect_lines(frame) -> np.ndarray: 
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

    # truncate entries above the halfway line on the image
    white_edges[0:int(imshape[0]/2), :] = 0
    yellow_edges[0:int(imshape[0]/2), :] = 0

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

def pix_in_range(x, y, shape):
    return y > 0 and x > 0 and y < shape[0] and x < shape[1]

def find_points_on_line(lines, imshape):
    pix_space = np.linspace(-1000, 1000, 2001, dtype=np.int16).reshape(2001, 1)
    line_pixels = []

    for i in range(len(lines)):

        rho = lines[i][0]
        theta = lines[i][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        xs = pix_space[:] * -b + x0
        ys = pix_space[:] * a + y0

        pixels = np.hstack((ys,xs)).astype(np.int16)

        pixels_in_range = np.array([pixel for pixel in pixels if pix_in_range(pixel[1], pixel[0], imshape)])
        unique_pixels = np.unique(pixels_in_range, axis=0)
        line_pixels.append(unique_pixels)

    return line_pixels

def pixels_to_world_points(depth_image, line_pixels):
    # takes pixels from the lines - list of np.ndarrays with shape (Nx2) where N is number of points in line
    # currently in form v, u

    line_point_list = []
    for i in range(len(line_pixels)): 
        depths = np.array([depth_image[pixel[0], pixel[1]] for pixel in line_pixels[i]])

        # something about this seems very off. Def check this. 
        x_pix = (line_pixels[i][ :, 1] - util.K_MAT[0,2]) / util.K_MAT[0,0]
        y_pix = (line_pixels[i][ :, 0] - util.K_MAT[1,2]) / util.K_MAT[1,1]
        line_points = np.array([x_pix, y_pix,depths]).T

        line_point_list.append(line_points)

    return line_point_list

def main():
    cap = cv2.VideoCapture("scene1_front.mp4") # scene 6 is a disaster...
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True).to(device)

    rand_start = random.randint(250,700)
    counter = 0
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video file.")
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more frames
            break
        # choosing a random frame in the video to begin analysis
        if counter < rand_start:
            counter+=1
            continue

        lines = detect_lines(frame) # returns L lines in a np.ndarray. Form Lx2 w/ axis 1 containing [rho, theta]
        if lines is None:
            continue
        line_pixels = find_points_on_line(lines, frame.shape[:2])
        depth_numpy = zoe.infer_pil(frame) # very slow :/ could batch them to speed it up?
        line_points = pixels_to_world_points(depth_numpy, line_pixels)
        util.show_line_points(line_points)
        util.visualize_stuff(frame, lines, depth_numpy)

if __name__ == '__main__':
    main()