#!/usr/bin/env python

import numpy as np
import rospy

# Computer Vision packages
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS Messages
from sensor_msgs.msgs import Image
from geometry_msgs.msg import Point, PoseArray
from visual_servoing.msg import ConeLocationPixel # custom message
from homography_transformer import HomographyTransformer

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation, lf_color_segmentation

class OrangeLineDetector():
    """
    A class for applying line detection algorithms to the real robot. 
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: <INSERT TRAJECTORY TOPIC> (PoseArray) : the trajectory of the detected line in the world frame.
    
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        
        self.traj_topic = rospy.get_param("~traj_topic", "/current_trajectory")
        self.traj_pub = rospy.Publisher(self.traj_topic, PoseArray, queue_size=10)

        self.homography = HomographyTransformer()
        print("We might want to overwrite homography transformer to be static")

        self.debug_pub = rospy.Publisher("/cone_debug_img", Image, queue_size=10)
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

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        lower_bound = (0, 100, 10)
        upper_bound = (50, 255, 255)
        template = [lower_bound, upper_bound]
        if self.LineFollower:
            bb = lf_color_segmentation(image, template)
        else:
            bb = cd_color_segmentation(image, template)
	    #cone_base = ((bb[0,0]+bb[1,0])/2, bb[1][1])
        cone_msg = ConeLocationPixel()
        cone_msg.u=(bb[0][0]+bb[1][0])/2
        cone_msg.v = bb[1][1]
        debug_img = cv2.rectangle(image, bb[0], bb[1], (0, 255, 0), 2) #adds box onto image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self.debug_pub.publish(debug_msg)
        self.cone_pub.publish(cone_msg)



if __name__ == '__main__':
    try:
        rospy.init_node('OrangeLineDetector', anonymous=True)
        OrangeLineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass