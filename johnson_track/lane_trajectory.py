#!/usr/bin/env python

from distutils.log import debug
import numpy as np
import rospy
from utils import LineTrajectory

# Computer Vision packages
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS Messages
from sensor_msgs.msgs import Image
from geometry_msgs.msg import Point32, PoseArray
from visual_servoing.msg import ConeLocationPixel # custom message
from homography_transformer import HomographyTransformer

# import your color segmentation algorithm; call this function in ros_image_callback!
from lane_detection import track_trajectory

class LaneTrajectory():
    """
    A class for applying line detection algorithms to the real robot. 
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: <INSERT TRAJECTORY TOPIC> (PoseArray) : the trajectory of the detected line in the world frame.
    
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_topic = rospy.get_param("~traj_topic", "/trajectory/current")
        self.traj_pub = rospy.Publisher(self.traj_topic, PoseArray, queue_size=10)

        self.homography = HomographyTransformer()

        self.debug_pub = rospy.Publisher("/traj_debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################

        self.trajectory.clear()
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        image_y, image_x = image.shape()

        trajectory_sides = track_trajectory(image)

        debug_img = image.copy()
        
        if trajectory_sides is not None: # todo: add something in lane_detection that allows this to be none
            trajectory_left = trajectory_sides[0]
            trajectory_right = trajectory_sides[1]
            for i in range(len(trajectory_left)):
                left_x, left_y = trajectory_left[i]
                right_x, right_y = trajectory_left[i]

                left_x_hom, left_y_hom = self.homography.transformUvToXy(left_x,left_y)
                right_x_hom, right_y_hom = self.homography.transformUvToXy(left_x,left_y)

                x_avg = (left_x_hom+right_x_hom)/2.0
                y_avg = (left_y_hom+right_y_hom)/2.0

                # debug_img = cv2.point(debug_img, point_pixels, 5, (255, 255, 0), 1)

                if 0 <= x_avg < image_x and 0 <= y_avg < image_y:
                    new_point = Point32(x_avg, y_avg, 0)
                    self.trajectory.addPoint(new_point)

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

        if (self.debug_pub.get_num_connections() > 0): self.debug_pub(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))



if __name__ == '__main__':
    try:
        rospy.init_node('LaneTrajectory', anonymous=True)
        LaneTrajectory()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass