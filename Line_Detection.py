import numpy as np
import cv2
import math

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
            cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
    return frame

def add_linesP(frame, linesP, color):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(frame, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)
    return frame

def main():
#   Hough lines...?
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

        low_sig, high_sig = 35,90 # 71,178

        white_pixels = cv2.inRange(frame_hls, (0,low_white, 0), (255,high_white, 255))
        yellow_pixels = cv2.inRange(frame_hsv, (low_h,low_s,low_v), (high_h,high_s,high_v))
        edges = cv2.Canny(frame, low_sig,high_sig)

        element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        # edges = cv2.erode(edges,element)
        edges = cv2.dilate(edges, element2)

        white_edges = cv2.bitwise_and(white_pixels, edges)
        yellow_edges = cv2.bitwise_and(yellow_pixels, edges)

        # truncate entries above the halfway line on the image
        white_edges[0:int(imshape[0]/2), :] = 0
        yellow_edges[0:int(imshape[0]/2), :] = 0
        
        # white_pixels[0:int(imshape[0]/2), :] = 0
        # yellow_pixels[0:int(imshape[0]/2), :] = 0


        lines_W = cv2.HoughLines(white_edges, 1, np.pi / 180, 100, None, 0, 0)
        lines_Y = cv2.HoughLines(yellow_edges, 1, np.pi / 180, 100, None, 0, 0)
        linesP_W = cv2.HoughLinesP(white_edges, 1, np.pi / 180, 100, None, 100, 25)
        linesP_Y = cv2.HoughLinesP(yellow_edges, 1, np.pi / 180, 100, None, 100, 25)

        frame = add_lines(frame, lines_W, (255,255,255))
        frame = add_lines(frame, lines_Y, (0,255,255))
        # frame = add_linesP(frame, linesP_W, (255,255,255))
        # frame = add_linesP(frame, linesP_Y, (255,255,0))


            
        scale = 0.33
        time_scale = 20
        cv2.imshow("edge_map", cv2.resize(edges, None, fx=scale, fy=scale))
        cv2.imshow("white_pixels", cv2.resize(white_pixels, None, fx=scale, fy=scale))
        # cv2.imshow("yellow_pixels", cv2.resize(yellow_pixels, None, fx=scale, fy=scale))
        cv2.imshow("White Edges", white_edges)
        cv2.imshow("Yellow Edges", yellow_edges)
        cv2.imshow("original", frame)
        # cv2.imshow("GT", cv2.resize(frame, None, fx=scale, fy=scale))

        cv2.waitKey(time_scale)

if __name__ == '__main__':
    main()