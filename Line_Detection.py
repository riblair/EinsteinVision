import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

THETA_HIGH = 1.8
THETA_LOW = 1.4

MAX_K = 5

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
    print(total_variance)
    if total_variance < 1:
        return None 

    for k in range(2, MAX_K+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(line_arr)
        kmeans_out.append(kmeans)
        score = silhouette_score(line_arr, kmeans.labels_)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    print(scores)
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

def show_images(edges, white_pixels, white_edges, yellow_edges, frame, lines_W, counter):
    scale = 0.33
    time_scale = 10
    cv2.imshow("edge_map", cv2.resize(edges, None, fx=scale, fy=scale))
    cv2.imshow("white_pixels", cv2.resize(white_pixels, None, fx=scale, fy=scale))
    # cv2.imshow("yellow_pixels", cv2.resize(yellow_pixels, None, fx=scale, fy=scale))
    cv2.imshow("White Edges", white_edges)
    cv2.imshow("Yellow Edges", yellow_edges)
    cv2.imshow("original", frame)
    # cv2.imshow("GT", cv2.resize(frame, None, fx=scale, fy=scale))
    if counter > 500 and counter % 50 == 0:
        graph_raw_lines(lines_W)
    cv2.waitKey(time_scale)

def main():
    counter = 0 
    cap = cv2.VideoCapture("scene1_front.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: # no more frames
            break
        imshape = frame.shape
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        low_white, high_white = 150, 255

        low_h, high_h = 10, 80
        low_s, high_s = 49,255
        low_v, high_v = 112,255

        low_sig, high_sig = 30,80 # 71,178

        white_pixels = cv2.inRange(frame_hls, (0,low_white, 0), (255,high_white, 255))
        yellow_pixels = cv2.inRange(frame_hsv, (low_h,low_s,low_v), (high_h,high_s,high_v))
        edges = cv2.Canny(frame, low_sig,high_sig)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

        white_edges = cv2.bitwise_and(white_pixels, edges)
        yellow_edges = cv2.bitwise_and(yellow_pixels, edges)

        # truncate entries above the halfway line on the image
        white_edges[0:int(imshape[0]/2), :] = 0
        yellow_edges[0:int(imshape[0]/2), :] = 0

        lines_W = cv2.HoughLines(white_edges, 1, np.pi / 180, 80, None, 0, 0)
        lines_Y = cv2.HoughLines(yellow_edges, 1, np.pi / 180, 80, None, 0, 0)
        # linesP_W = cv2.HoughLinesP(white_edges, 1, np.pi / 180, 100, None, 100, 25)
        # linesP_Y = cv2.HoughLinesP(yellow_edges, 1, np.pi / 180, 100, None, 100, 25)

        frame = add_lines(frame, lines_W, (255,255,255))
        frame = add_lines(frame, lines_Y, (0,255,255))
        # frame = add_linesP(frame, linesP_W, (255,255,255))
        # frame = add_linesP(frame, linesP_Y, (255,255,0))
        lines_W = clean_data(lines_W)
        lines_Y = clean_data(lines_Y)

        all_lines = lines_W + lines_Y
        # print(all_lines)
        if len(all_lines):
            kmeans = cluster_lines(all_lines)
            if kmeans is None:
                pass # hmm
            if counter > 100 and counter % 50 == 0:
                graph_clusters(kmeans, all_lines)

        show_images(edges, white_pixels, white_edges, yellow_edges, frame, lines_W, counter)
            
        counter +=1
if __name__ == '__main__':
    main()