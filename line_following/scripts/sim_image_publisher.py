#!/usr/bin/env python

from distutils.log import debug
import numpy as np
import rospy

# Computer Vision packages
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS Messages
from sensor_msgs.msg import Image

class ImagePublisher():
    def __init__(self):
        self.pub_topic = "/test_img"
        self.image_pub = rospy.Publisher(self.pub_topic, Image, queue_size=1)
        self.bridge= CvBridge()
        image = cv2.imread("./test_pics/7.png")
        image_message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        print(image_message)
        self.image_pub.publish(image_message)

if __name__ == "__main__":
  #    try:
    rospy.init_node('photo_publisher', anonymous=True)
    ImagePublisher()
    rospy.spin()
#    except:
#        print("oops")
#        pass
