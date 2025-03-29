import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import torch

LOW_WHITE, HIGH_WHITE = 150, 255

LOW_H, HIGH_H = 10, 80
LOW_S, HIGH_S = 49,255
LOW_V, HIGH_V = 112,255

LOW_SIG, HIGH_SIG = 30,80

THETA_HIGH = 1.8
THETA_LOW = 1.4

MAX_K = 5



# TODO - implement monocular depth estimationn via ZoeDepth

# Key improvements that need to be made
# Reject Thetas in another range (ones that are too vertical)
# Implement 1 lane solution
# Reject data that is too far away from detected lines
# Implement some consistency between frames perhaps 

def clean_data(lines):
    cleaned_lines = []
    if lines is not None:
        for line in lines:
            if line[0][1] > 1.8 or line[0][1] < 1.4:
                cleaned_lines.append(line)
    return cleaned_lines

def cluster_lines(lines):
    line_arr = np.array(lines).squeeze(1)
    scores = []
    kmeans_out = []
    if len(line_arr) < 7: # not enough data to be clustering? NOT ELEGANT, FIX!
        return None
    total_variance = np.var(line_arr, axis=0).sum()
    # print(total_variance)
    if total_variance < 1:
        return None 

    for k in range(2, MAX_K+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(line_arr)
        kmeans_out.append(kmeans)
        score = silhouette_score(line_arr, kmeans.labels_)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    # print(scores)
    return kmeans_out[best_k-2]

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
            if theta < 1.8 and theta > 1.4: # lines must be greater than 103 deg OR less than 80 deg 
                cv2.line(frame, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
            else:
                cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
    return frame

def draw_lane_lines(frame, kmeans):
    lines = kmeans.cluster_centers_
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

def add_linesP(frame, linesP, color):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)
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
    colors = ['Red','Green', 'Blue', 'Orange', 'Black']

    for i in range(cluster_num):
        rhos = []
        thetas = []
        for j in range(len(lines)):
            label = kmeans.labels_[j]
            if label == i:
                rhos.append(float(lines[j][0][0]))
                thetas.append(float(lines[j][0][1]))
        plt.scatter(rhos, thetas, c=colors[i])
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

def detect_lines(frame) -> np.ndarray: 
    imshape = frame.shape

    """Convert frame to hsv for detecting yellow pixels, and hls for detecting white pixels """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    """ Color Threshholding and edge detection """
    white_pixels = cv2.inRange(frame_hls, (0,LOW_WHITE, 0), (255,HIGH_WHITE, 255))
    yellow_pixels = cv2.inRange(frame_hsv, (LOW_H,LOW_S,LOW_V), (HIGH_H,HIGH_S,HIGH_V))
    edges = cv2.Canny(frame, LOW_SIG, HIGH_SIG)
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
    # show_images(edges, white_pixels, white_edges, yellow_edges, frame, lines_W)
    return None

def map_to_range(arr, min_val, max_val, new_min, new_max):
    return [new_min + (x - min_val) * (new_max - new_min) / (max_val - min_val) for x in arr]

def main():
    cap = cv2.VideoCapture("scene1_front.mp4") # scene 6 is a disaster...
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = "isl-org/ZoeDepth"
    # Zoe_N
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    zoe = model_zoe_n.to(device)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more frames
            break

        # lines = detect_lines(frame) # returns L lines in a np.ndarray. Form Lx2 w/ axis 1 containing [rho, theta]
        # if not lines:
        #     continue

        depth_numpy = zoe.infer_pil(frame)

        remapped = map_to_range(depth_numpy, np.min(depth_numpy), np.max(depth_numpy), 0, 255)
        remapped = np.array(remapped, dtype=np.uint8)

        cv2.imshow('a', cv2.resize(remapped, remapped.shape/2))

        cv2.waitKey(27)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # exit(1)

if __name__ == '__main__':
    main()