#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

HORIZON = 0.4

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

def intersection_to_horizontal(image, lines,horizontal_perc):
    midline = image.shape[1] / 2.0
    closest_line = None
    closest_dist_midline = float('inf')

    for line in lines:
        dist_midline = np.abs(intersection(image,line,horizontal_perc)[0] - midline)
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


def hough(image,og_image,horizontal_perc):
    image_copy = og_image.copy()

    threshold = 150
    lines_pos_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    lines_neg_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)
    horizontal_line = np.array([[[int(image_copy.shape[0]*.45),np.pi/2]]])

    threshold = 150
    while lines_pos_slope is None:
        threshold = int(threshold/2.0)
        print("pos")
        print(threshold)
        lines_pos_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    
    viz_img = line_visualizer(image_copy,lines_pos_slope,(255,255,255))
    cv2.imwrite('viz' + '.png',viz_img)

    threshold = 150
    while lines_neg_slope is None:
        print("neg")
        print(threshold)
        threshold = int(threshold/2.0)
        lines_neg_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)

    #image_copy = line_visualizer(image_copy, lines_pos_slope,(255.0,0.0,0.0)) # blue
    #image_copy = line_visualizer(image_copy, lines_neg_slope,(0.0,255.0,0.0)) # green
    image_copy = line_visualizer(image_copy,horizontal_line,(0.0,0.0,255.0)) # red -- horizontal

    intersections_pos_slope = intersection_to_horizontal(image_copy, lines_pos_slope,horizontal_perc)
    intersections_neg_slope = intersection_to_horizontal(image_copy, lines_neg_slope,horizontal_perc)

    image_copy = line_visualizer(image_copy, np.array([intersections_pos_slope]),(0.0,255.0,255.0)) # yellow
    image_copy = line_visualizer(image_copy, np.array([intersections_neg_slope]),(0.0,255.0,255.0)) # yellow

    image_copy = cv2.circle(image_copy,(0,int(image_copy.shape[0]*.4)),radius=0,color=(0,0,255),thickness=-1)

    cv2.imwrite('hough_lines' + '.png',image_copy)

    return [intersections_pos_slope,intersections_neg_slope]

def trajectory_rails(image,left_line, right_line, steps):
    left_points = []
    right_points = []
    for step in steps:
        left_points.append(intersection(image,left_line,step))
        right_points.append(intersection(image,right_line,step))
    return [left_points,right_points]

def track_trajectory(image):
    blur = gaussian_blur(color2grayscale(image))
    canny_edge_image = canny_edges(blur)
    trajectory_image = image.copy()

    
    # BOTTOM
    # going from 0.5 to 1
    # pinch point at 0.6
    # collect data at 0.5, 0.6, 0.7, 0.8, 0.9
    bottom_edges = canny_edge_image.copy()
    bottom_edges = cv2.rectangle(bottom_edges, (0,0), (bottom_edges.shape[1],int(bottom_edges.shape[0]*0.5)), color=0.0, thickness=-1)
    
    # cv2.imwrite('top_edges' + '.png',top_edges)
    horizontal_percentage = 0.6
    intersections_pos_line,intersections_neg_line = hough(bottom_edges,image,horizontal_percentage)
    bottom_steps = [0.9,0.8,0.7,0.6,0.5]
    bottom_trajectory_points = trajectory_rails(image,intersections_pos_line,intersections_neg_line,bottom_steps)

    # TOP
    # going from 0.4 (horizon) to 0.5
    # pinch point at 0.45
    # collect data at 0.42, 0.43, ... 0.49
    top_edges = canny_edge_image.copy()
    top_edges = cv2.rectangle(top_edges, (0,0), (top_edges.shape[1],int(top_edges.shape[0]*HORIZON)), color=0.0, thickness=-1)
    top_edges = cv2.rectangle(top_edges,(0,int(top_edges.shape[0]*0.5)), (top_edges.shape[1],top_edges.shape[0]), color=0.0, thickness=-1)
    
    horizontal_percentage = 0.45
    intersections_pos_line,intersections_neg_line = hough(top_edges,image,horizontal_percentage)
    top_steps = [0.49,0.48,0.47,0.46,0.45,0.44,0.43,0.42]
    top_trajectory_points = trajectory_rails(image,intersections_pos_line,intersections_neg_line,top_steps)

    trajectory_image = line_visualizer(trajectory_image,np.array([intersections_neg_line]),(255,0,0))
    cv2.imwrite('top_edges' + '.png',top_edges)
    cv2.imwrite('trajectory_viz' + '.png',trajectory_image)



    trajectory_points = [bottom_trajectory_points[0] + top_trajectory_points[0],bottom_trajectory_points[1] + top_trajectory_points[1]]

    
    for point in trajectory_points[0]:
        trajectory_image = cv2.circle(trajectory_image,point,radius=1,color=(0,0,255),thickness=-1)
    for point in trajectory_points[1]:
        trajectory_image = cv2.circle(trajectory_image,point,radius=1,color=(0,0,255),thickness=-1)
    cv2.imwrite('trajectory_image' + '.png',trajectory_image)

    return trajectory_points

    # example:
    # [[[-99, 338], [-27, 300], [42, 263], [114, 225], [183, 188]], [[756, 338], [662, 300], [571, 263], [477, 225], [385, 188]]]
    # [left, right]
    # nearest to the bottom to furthest from bottom
    # note: can have neg values!!!


def intersection_at_pinch(image,lines_pos_slope,lines_neg_slope,pinch_percentage,midline):
    closest_line_left = None
    closest_dist_midline_left = float('inf')

    closest_line_right = None
    closest_dist_midline_right = float('inf')

    for line in lines_pos_slope:
        dist_midline = intersection(image,line,pinch_percentage)[0] - midline
        if dist_midline < 0 and (-1*dist_midline) < closest_dist_midline_left: # left
            closest_dist_midline_left = abs(dist_midline)
            closest_line_left = line
        if dist_midline > 0 and dist_midline < closest_dist_midline_right: # right
            closest_dist_midline_right = dist_midline
            closest_line_right = line
    
    for line in lines_neg_slope:
        dist_midline = intersection(image,line,pinch_percentage)[0] - midline
        if dist_midline < 0 and (-1*dist_midline) < closest_dist_midline_left: # left
            closest_dist_midline_left = abs(dist_midline)
            closest_line_left = line
        if dist_midline > 0 and dist_midline < closest_dist_midline_right: # right
            closest_dist_midline_right = dist_midline
            closest_line_right = line

    return [closest_line_left, closest_line_right]

def hough_section(image, top_bound, bottom_bound, pinch, base_threshold, midline, viz_image):
    section = image.copy()

    section = cv2.rectangle(section, (0,0), (section.shape[1],int(section.shape[0]*top_bound)), color=0.0, thickness=-1)  # cut off top
    section = cv2.rectangle(section,(0,int(section.shape[0]*bottom_bound)), (section.shape[1],section.shape[0]), color=0.0, thickness=-1) # cut off bottom

    cv2.imwrite('section.png',section)

    threshold = base_threshold
    lines_pos_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    lines_neg_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)
    horizontal_line = np.array([[[int(section.shape[0]*.43),np.pi/2]]])

    threshold = base_threshold
    while lines_pos_slope is None:
        print("pos")
        print(threshold)
        threshold = int(threshold/2.0)
        lines_pos_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    
    threshold = base_threshold
    while lines_neg_slope is None:
        print("neg")
        print(threshold)
        threshold = int(threshold/2.0)
        lines_neg_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)


    viz_image2 = viz_image.copy()
    
    viz_image = cv2.circle(viz_image,(int(viz_image.shape[1]*.5),int(viz_image.shape[0]*.5)),radius=0,color=(0,0,255),thickness=-1)
    viz_image = line_visualizer(viz_image, lines_pos_slope,(255.0,0.0,0.0)) # blue
    viz_image = line_visualizer(viz_image, lines_neg_slope,(0.0,255.0,0.0)) # green
    viz_image = line_visualizer(viz_image, horizontal_line,(0.0,0.0,255.0)) # green



    cv2.imwrite('section_traj.png',viz_image)

    closest_line_left, closest_line_right  = intersection_at_pinch(image,lines_pos_slope,lines_neg_slope,pinch,midline)
    viz_image2 = line_visualizer(viz_image2, np.array([closest_line_left]),(255.0,0.0,0.0)) # blue
    viz_image2 = line_visualizer(viz_image2, np.array([closest_line_right]),(0.0,255.0,0.0)) # green
    cv2.imwrite('section_traj2.png',viz_image2)

    return (closest_line_left, closest_line_right)

def get_trajectory(image):
    blur = gaussian_blur(color2grayscale(image))
    canny_edge_image = canny_edges(blur)
    trajectory_image = image.copy()

    
    # 0.4 --> 1
    # 0.4 --> 0.45: pinch point at 0.43 : threshold 50 [.44,.43,.43,.41]
    # 0.45 --> 0.5: pinch point at 0.48 : threshold 50 [.49,.48,.47,.46,.45]
    # 0.5 --> 0.7: pinch point at 0.6 : threshold 150 [.7, .6, .5]
    # 0.7 --> 1.0: pinch point at 0.8 : threshold 150 [.9, .8]

    all_left_points = []
    all_right_points = []
    
    closest_line_left, closest_line_right = hough_section(canny_edge_image,0.5,1.0, 0.6, 150, image.shape[1] / 2.0, image.copy())
    steps = [.9,.8,.7, .65, .6, .55, .5]
    left_points,right_points = trajectory_rails(image, closest_line_left,closest_line_right,steps)
    all_left_points.extend(left_points)
    all_right_points.extend(right_points)

    new_midline = (all_left_points[-1][0] + all_right_points[-1][0])/2.0

    closest_line_left, closest_line_right = hough_section(canny_edge_image,0.45,0.5, 0.48, 150, new_midline, image.copy())
    steps = [.49,.48,.47,.46,.45]
    left_points,right_points = trajectory_rails(image, closest_line_left,closest_line_right,steps)
    if (np.abs(all_left_points[-1][0] - left_points[0][0]) < 50):
        all_left_points.extend(left_points)
    if (np.abs(all_right_points[-1][0] - right_points[0][0]) < 50):
        all_right_points.extend(right_points)

    new_midline = (all_left_points[-1][0] + all_right_points[-1][0])/2.0

    closest_line_left, closest_line_right = hough_section(canny_edge_image,0.4,0.45, 0.43, 50, new_midline, image.copy())
    steps = [.44,.43,.42,.41]
    left_points,right_points = trajectory_rails(image, closest_line_left,closest_line_right,steps)
    if (np.abs(all_left_points[-1][0] - left_points[0][0]) < 50):
        all_left_points.extend(left_points)
    if (np.abs(all_right_points[-1][0] - right_points[0][0]) < 50):
        all_right_points.extend(right_points)

    print(all_left_points)
    print(all_right_points)

    trajectory_points = [all_left_points, all_right_points]

    for point in trajectory_points[0]:
        trajectory_image = cv2.circle(trajectory_image,point,radius=1,color=(0,0,255),thickness=-1)
    for point in trajectory_points[1]:
        trajectory_image = cv2.circle(trajectory_image,point,radius=1,color=(0,0,255),thickness=-1)
    #cv2.imwrite('trajectory_image' + '.png',trajectory_image)

    return trajectory_image
    #return trajectory_points
    # example:
    # [[[-99, 338], [-27, 300], [42, 263], [114, 225], [183, 188]], [[756, 338], [662, 300], [571, 263], [477, 225], [385, 188]]]
    # [left, right]
    # nearest to the bottom to furthest from bottom
    # note: can have neg values!!!


def main():
    for i in range(1,18): # (1,18)
        image_path = r'C:\Users\shrey\OneDrive\Desktop\track' + str(i) + '.png'
        image = cv2.imread(image_path)

        trajectory_image = get_trajectory(image)
        cv2.imwrite('trajectory_image' + str(i) + '.png',trajectory_image)
        #track_trajectory(image)


        # cv2.imwrite('blur.png',blur)
        #cv2.imwrite('final_edges_' + str(i) + '.png',final_edges)
        #cv2.imwrite('hough_lines_' + str(i) + '.png',disp_image)

if __name__ == "__main__":
    main()