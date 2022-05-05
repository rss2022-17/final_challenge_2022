#!/usr/bin/env python

from distutils.log import debug
import queue
import numpy as np
import rospy
from utils import LineTrajectory

# Computer Vision packages
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS Messages
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32, PoseArray
from visual_servoing.msg import ConeLocationPixel # custom message
from homography_transformer import HomographyTransformer

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation2 import lf_color_segmentation

class OrangeLineDetector2():
    """
    A class for applying line detection algorithms to the real robot. 
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: <INSERT TRAJECTORY TOPIC> (PoseArray) : the trajectory of the detected line in the world frame.
    
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_topic = "/linecenter"
        self.traj_pub = rospy.Publisher(self.traj_topic, Point32, queue_size=1)
        self.image_topic = rospy.get_param("~image_topic", "zed/zed_node/rgb/image_rect_color")
        self.line_follower_state_topic = rospy.get_param("~active_state", "/line_follower")


        self.last_turn_left = None
        self.turn_state_pub = rospy.Publisher("/turn_state", Bool, queue_size=1)
        self.turn_left_pub = rospy.Publisher("/turn_left", Bool, queue_size=1)

        self.active_state = rospy.get_param("~start_active", False)

        self.homography = HomographyTransformer()

        self.state_sub = rospy.Subscriber(self.line_follower_state_topic, Bool, self.state_callback)
        self.debug_pub = rospy.Publisher("/traj_debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.trajectory.addPoint(Point32(0,0,0))

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

        # If the state machine hasn't activated, skip over the callback
        if not self.active_state: return

        #self.trajectory.clear()
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        lower_bound = (5, 190, 100)
        upper_bound = (40, 255, 255)
        template = [lower_bound, upper_bound]

        trajectory_point = lf_color_segmentation(image, template, pct=0.5)
        #debug_img = image.copy()
        msg = Point32()
        msg.x, msg.y = self.homography.transformUvToXy(trajectory_point[0], trajectory_point[1])
        self.traj_pub.publish(msg)

        #if (self.debug_pub.get_num_connections() > 0): self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

    # def state_callback(self, msg):
    #     self.active_state = msg.data



if __name__ == '__main__':
    try:
        rospy.init_node('OrangeLineDetector2', anonymous=True)
        OrangeLineDetector2()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass