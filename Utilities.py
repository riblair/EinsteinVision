import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

COLORS = ['Red','Green', 'Blue', 'Orange', 'Black']

LOW_WHITE, HIGH_WHITE = 150, 255
LOW_H, HIGH_H = 10, 80
LOW_S, HIGH_S = 49,255
LOW_V, HIGH_V = 112,255
LOW_SIG, HIGH_SIG = 30,80

MAX_CLUSTERS = 5

K_MAT = np.array([[1594.7,         0,    655.3],
                  [     0,    1607.7,    414.4],
                  [     0,         0,        1]])

HIGH_THETA = 1.8
LOW_THETA = 1.4

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

def show_line_points(line_points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    for i in range(len(line_points)):
        ax.scatter(line_points[i][:,0],line_points[i][:,1],line_points[i][:,2], c=COLORS[i])

def visualize_stuff(frame, lines, depth_map):
    def map_to_range(arr, min_val, max_val, new_min, new_max):
            return [new_min + (x - min_val) * (new_max - new_min) / (max_val - min_val) for x in arr]
    frame = draw_lane_lines(frame, lines)
    remapped = map_to_range(depth_map, np.min(depth_map), np.max(depth_map), 0, 255)
    remapped = np.array(remapped, dtype=np.uint8)
    cv2.imshow('frame', frame)
    cv2.imshow('depth', remapped)
    cv2.waitKey(100)
    plt.show()
