#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

def color2grayscale(image):
    """
    converts cv2 image from color --> grayscale

    Args:
        image (cv2 image): image in color

    Returns:
        cv2 image: image in grayscale
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image):
    x_kernel = 7
    y_kernel = 7
    return cv2.GaussianBlur(image,(x_kernel, y_kernel),0)

def canny_edges(image):
    canny_param_1 = 50
    canny_param_2 = 150
    edges = cv2.Canny(image,canny_param_1,canny_param_2)
    return edges

def line_visualizer(image,lines,color):
    for line in range(0,len(lines)):
        rho = lines[line,0,0]
        theta = lines[line,0,1]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        pt1_x = int(np.rint(x0 + 1000*(-b)))
        pt1_y = int(np.rint(y0 + 1000*(a)))
        pt2_x = int(np.rint(x0 - 1000*(-b)))
        pt2_y = int(np.rint(y0 - 1000*(a)))

        pt1 = (pt1_x,pt1_y)
        pt2 = (pt2_x,pt2_y)

        image = cv2.line(image,pt1,pt2,color,2)
    return image

def intersection_to_horizontal(image, lines):
    midline = image.shape[1] / 2.0
    closest_line = None
    closest_dist_midline = float('inf')

    for line in lines:
        dist_midline = np.abs(intersection(image,line,0.6)[0] - midline)
        if dist_midline < closest_dist_midline:
            closest_dist_midline = dist_midline
            closest_line = line

    return closest_line

def intersection(image, line1, y_placement):
    # https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    rho1, theta1 = line1[0]
    rho2, theta2 = int(image.shape[0]*y_placement), np.pi/2 # horizontal
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))

    return [x0, y0]


def hough(image,og_image):
    image_copy = og_image.copy()

    threshold = 150

    lines_pos_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    lines_neg_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)
    horizontal_line = np.array([[[int(image_copy.shape[0]*.6),np.pi/2]]])

    while lines_pos_slope is None:
        threshold = threshold/3
        lines_pos_slope = cv2.HoughLines(image,1, np.pi/180.0,50,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    
    threshold = 150
    if lines_neg_slope is None:
        threshold = threshold/3
        lines_neg_slope = cv2.HoughLines(image,1, np.pi/180.0,50,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)


    #image_copy = line_visualizer(image_copy, lines_pos_slope,(255.0,0.0,0.0)) # blue
    #image_copy = line_visualizer(image_copy, lines_neg_slope,(0.0,255.0,0.0)) # green
    image_copy = line_visualizer(image_copy,horizontal_line,(0.0,0.0,255.0)) # red -- horizontal

    intersections_pos_slope = intersection_to_horizontal(image_copy, lines_pos_slope)
    intersections_neg_slope = intersection_to_horizontal(image_copy, lines_neg_slope)

    image_copy = line_visualizer(image_copy, np.array([intersections_pos_slope]),(0.0,255.0,255.0)) # yellow
    image_copy = line_visualizer(image_copy, np.array([intersections_neg_slope]),(0.0,255.0,255.0)) # yellow

    image_copy = cv2.circle(image_copy,(0,int(image_copy.shape[0]*.4)),radius=0,color=(0,0,255),thickness=-1)


    return [intersections_pos_slope,intersections_neg_slope]

def trajectory_rails(image,left_line, right_line):
    left_points = []
    right_points = []
    for i in range(9,4,-1):
        step = i/10.0
        left_points.append(intersection(image,left_line,step))
        right_points.append(intersection(image,right_line,step))
    return [left_points,right_points]

def track_trajectory(image):
    blur = gaussian_blur(color2grayscale(image))
    final_edges = canny_edges(blur)
    final_edges = cv2.rectangle(final_edges, (0,0), (final_edges.shape[1],int(final_edges.shape[0]*0.39)), color=0.0, thickness=-1)
    intersections_pos_line,intersections_neg_line = hough(final_edges,image)
    trajectory_points = trajectory_rails(image,intersections_pos_line,intersections_neg_line)
    
    print(trajectory_points)
    return trajectory_points
    # example:
    # [[[-99, 338], [-27, 300], [42, 263], [114, 225], [183, 188]], [[756, 338], [662, 300], [571, 263], [477, 225], [385, 188]]]
    # [left, right]
    # nearest to the bottom to furthest from bottom
    # note: can have neg values!!!

def main():
    for i in range(1,18): # (1,18)
        image_path = r'C:\Users\shrey\OneDrive\Desktop\track' + str(i) + '.png'
        image = cv2.imread(image_path)

        track_trajectory(image)
        # blur =gaussian_blur(color2grayscale(image))

        # final_edges = canny_edges(blur)
        # final_edges = cv2.rectangle(final_edges, (0,0), (final_edges.shape[1],int(final_edges.shape[0]*0.39)), color=0.0, thickness=-1)

        # hough_lines = hough(final_edges,image)

        # cv2.imwrite('blur.png',blur)
        # cv2.imwrite('final_edges.png',final_edges)
        # cv2.imwrite('hough_lines_' + str(i) + '.png',hough_lines)

if __name__ == "__main__":
    main()