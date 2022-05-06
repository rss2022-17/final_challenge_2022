#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
HORIZON = 0.4

def color2grayscale(image):
    """
    converts cv2 image from color --> grayscale

    Args:
        image (cv2 image): image in color

    Returns:
        cv2 image: image in grayscale
    """
    gray_im = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    white_im = 255*(gray_im > 170)

    # image_path = r'C:\Users\shrey\OneDrive\Desktop\white_image.png'
    # cv2.imwrite(image_path,np.uint8(white_im))

    return np.uint8(white_im)
def dilate(image):
    kernel=np.ones((5,5), np.uint8)
    image_dilated=cv2.dilate(image,kernel,iterations=1)
    return image_dilated
def gaussian_blur(image):
    x_kernel = 23
    y_kernel = 23
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
    threshold = 150
    lines_pos_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    lines_neg_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)

    threshold = 150
    while lines_pos_slope is None:
        threshold = int(threshold/2.0)
        if threshold==0:
            return None, None #return none if threshold doesn't detect anything with 0
        lines_pos_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    
    threshold = 150
    while lines_neg_slope is None:
        threshold = int(threshold/2.0)
        if threshold==0:#return none if threshold doesn't detect anything with 0
            return None, None
        lines_neg_slope = cv2.HoughLines(image,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)

    intersections_pos_slope = intersection_to_horizontal(og_image, lines_pos_slope,horizontal_perc)
    intersections_neg_slope = intersection_to_horizontal(og_image, lines_neg_slope,horizontal_perc)

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
    
    # BOTTOM
    # going from 0.5 to 1
    # pinch point at 0.6
    # collect data at 0.5, 0.6, 0.7, 0.8, 0.9
    bottom_edges = canny_edge_image.copy()
    bottom_edges = cv2.rectangle(bottom_edges, (0,0), (bottom_edges.shape[1],int(bottom_edges.shape[0]*0.5)), color=0.0, thickness=-1)
    
    horizontal_percentage = 0.6
    intersections_pos_line,intersections_neg_line = hough(bottom_edges,image,horizontal_percentage)
    if intersections_pos_line is None:#if no lines detected return a single point at 0,0 in the trajectory
        return np.array([0,0])
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
    if intersections_pos_line is None:#if no lines detected return a single point at 0,0 in the trajectory
        return np.array([0,0])
    #top_steps = [0.49,0.48,0.47,0.46,0.45,0.44,0.43,0.42]
    top_steps = [0.48,0.46,0.44,0.42]
    top_trajectory_points = trajectory_rails(image,intersections_pos_line,intersections_neg_line,top_steps)

    trajectory_points = [bottom_trajectory_points[0] + top_trajectory_points[0],bottom_trajectory_points[1] + top_trajectory_points[1]]

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

    threshold = base_threshold
    lines_pos_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    lines_neg_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)

    threshold = base_threshold
    while lines_pos_slope is None:
        threshold = int(threshold/2.0)

        if not threshold: return None, None

        lines_pos_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=0,max_theta=np.pi*0.4)
    
    threshold = base_threshold
    while lines_neg_slope is None:
        threshold = int(threshold/2.0)

        if not threshold: return None, None

        lines_neg_slope = cv2.HoughLines(section,1, np.pi/180.0,threshold,srn=0,stn=0,min_theta=np.pi*0.6,max_theta=np.pi)

    closest_line_left, closest_line_right  = intersection_at_pinch(image,lines_pos_slope,lines_neg_slope,pinch,midline)

    return (closest_line_left, closest_line_right)

def get_trajectory(image):

    #Blur image to reduce noise
    #dilation = dilate(color2grayscale(image))
    blur = gaussian_blur(image)
    #blur = dilation
    #Gets edges from image 
    canny_edge_image = canny_edges(blur)

    #Make a copy for base image
    trajectory_image = image.copy()


    # 0.4 --> 1
    # 0.4 --> 0.45: pinch point at 0.43 : threshold 50 [.44,.43,.43,.41]
    # 0.45 --> 0.5: pinch point at 0.48 : threshold 50 [.49,.48,.47,.46,.45]
    # 0.5 --> 0.7: pinch point at 0.6 : threshold 150 [.7, .6, .5]
    # 0.7 --> 1.0: pinch point at 0.8 : threshold 150 [.9, .8]

    all_left_points = []
    all_right_points = []
    
    # ------------------- BOTTOM -------------------
    closest_line_left, closest_line_right = hough_section(canny_edge_image,0.45,1.0, 0.6, 150, image.shape[1] / 2.0, image.copy())

    # trajectory_image = line_visualizer(trajectory_image, np.array([closest_line_left,closest_line_right]),(255,0,0))
    # image_path = r'C:\Users\shrey\OneDrive\Desktop\aaa.png'
    # cv2.imwrite(image_path,trajectory_image)

    if closest_line_left is None or closest_line_right is None: return None #np.array([0, 0])

    # steps = [.9,.8,.7, .65, .6, .55, .5]
    # steps = [.9,.8,.7,.6,.5]
    steps = [.45]
    left_points,right_points = trajectory_rails(image, closest_line_left,closest_line_right,steps)
    all_left_points.extend(left_points)
    all_right_points.extend(right_points)

    #new_midline = (all_left_points[-1][0] + all_right_points[-1][0])/2.0

    # ------------------- TOP -------------------
    closest_line_left, closest_line_right = hough_section(canny_edge_image,0.45,0.5, 0.48, 150, image.shape[1] / 2.0, image.copy())   
    if closest_line_left is None or closest_line_right is None: 
        return [all_left_points, all_right_points] #np.array([0, 0])


    # steps = [.49,.48,.47,.46,.45]
    # steps = [.47,.45]
    steps = [.43]
    left_points,right_points = trajectory_rails(image, closest_line_left,closest_line_right,steps)
    if (np.abs(all_left_points[-1][0] - left_points[0][0]) < 100) and (np.abs(all_right_points[-1][0] - right_points[0][0]) < 100):
        all_left_points.extend(left_points)
        all_right_points.extend(right_points)
 

    # new_midline = (all_left_points[-1][0] + all_right_points[-1][0])/2.0

    # closest_line_left, closest_line_right = hough_section(canny_edge_image,0.4,0.45, 0.43, 50, new_midline, image.copy())
    # if closest_line_left is None or closest_line_right is None: return None #np.array([0, 0])
    # steps = [.44,.43,.42,.41]
    # left_points,right_points = trajectory_rails(image, closest_line_left,closest_line_right,steps)
    # if (np.abs(all_left_points[-1][0] - left_points[0][0]) < 50) and (np.abs(all_right_points[-1][0] - right_points[0][0]) < 50):
    #     all_left_points.extend(left_points)
    #     all_right_points.extend(right_points)

    trajectory_points = [all_left_points, all_right_points]

    return trajectory_points
    # example:
    # [[[-99, 338], [-27, 300], [42, 263], [114, 225], [183, 188]], [[756, 338], [662, 300], [571, 263], [477, 225], [385, 188]]]
    # [left, right]
    # nearest to the bottom to furthest from bottom
    # note: can have neg values!!!

def get_trajectory2(image):
    debug_pub = rospy.Publisher("/hough_debug", Image, queue_size=10)

    src = image
    #Blur image to reduce noise
    blur = gaussian_blur(color2grayscale(image))

    #Gets edges from image 
    canny_edge_image = canny_edges(blur)

    cropped =  canny_edge_image[image.shape[0]//2:,:]
    cropped = canny_edges(blur)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(canny_edge_image, cv2.COLOR_GRAY2BGR)
    cdst2 = cv2.cvtColor(canny_edge_image, cv2.COLOR_GRAY2BGR) 
    cdstP = np.copy(cdst)
    
    left = cv2.HoughLines(cropped, 1, np.pi / 180, 150, None, 0, 0)
    right = cv2.HoughLines(cropped, 1, np.pi / 180, 150, None, 0, 0)
    
    if left is not None:
        for i in range(0, len(left)):
            rho = left[i][0][0]
            theta = left[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a))) #Add back half the screen for the lines
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    if right is not None:
        for i in range(0, len(right)):
            rho = right[i][0][0]
            theta = right[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a))) #Add back half the screen for the lines
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst2, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)


    linesP = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, None, 150, 10)
    
    x_list = []
    y_list = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            x_list.append((l[0] + l[2])/2)
            y_list.append((l[1] + l[3])/2)

    else:
        return None

    xmean = int(np.mean(x_list))
    ymean = int(np.mean(y_list))
    point = (xmean, ymean)



    return [xmean, ymean, cdstP]
    cv2.circle(src,point,radius=5,color=(0,255,0),thickness=-1)
    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cv2.waitKey(0)

    
    return 0
def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    for i in ['_test']:

        image_path = r'C:\Users\shrey\OneDrive\Desktop\track'+str(i)+'.png'
        image = cv2.imread(image_path)

        trajectory_image = image.copy()

        trajectory_points = get_trajectory(image)
        # print(trajectory_points)

        for point in trajectory_points[0]:
                trajectory_image = cv2.circle(trajectory_image,point,radius=1,color=(0,0,255),thickness=-1)
        for point in trajectory_points[1]:
                trajectory_image = cv2.circle(trajectory_image,point,radius=1,color=(0,0,255),thickness=-1)

        image_path = r'C:\Users\shrey\OneDrive\Desktop\aaa'+str(i)+'.png'

        # cv2.imwrite(image_path,trajectory_image)


        # cv2.imwrite('blur.png',blur)
        #cv2.imwrite('final_edges_' + str(i) + '.png',final_edges)
        #cv2.imwrite('hough_lines_' + str(i) + '.png',disp_image)

if __name__ == "__main__":
    img0 = cv2.imread('aaa.png')
    # img = gaussian_blur(img0)
    # show_image(img)
    # img = canny_edges(img)
    # show_image(img)
    trajectory_image = img0.copy()
    get_trajectory2(img0)
    #main()
    pass
