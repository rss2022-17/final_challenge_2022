#!/usr/bin/env python
import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
#from detector import StopSignDetector
from std_msgs.msg import Bool, Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped

class StopSignControl():

    def __init__(self):
        self.bbox_sub = rospy.Subscriber("stop_sign_bbox", Float32MultiArray, self.bbox_callback)
        self.sign_in_range= rospy.Publisher("stop_detected", Bool, queue_size = 1)
        #self.image_sub = rospy.Subscriber("stop_sign_debug", Image, self.image_callback)

        self.drive_topic = rospy.get_param("~turn_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_1")
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)

        self.time_to_wake = None
        self.state_active = False
        self.state_sub = rospy.Subscriber("stop_state", Bool, self.state_callback)

        
        #Currently just a guess for the transformation/distance for testing purposes


    """
    Callback method for bbox's
    """
    def bbox_callback(self, data):

        if self.time_to_wake is not None:

            if rospy.get_time() <= self.time_to_wake:
                return
            else:
                self.time_to_wake = None

        bbox = data.data
        #If no bounding box was found a.k.a no stop sign detected
        if len(bbox)==0:
            self.sign_in_range.publish(False)
        
        else:
            #dist_to_stop = self.get_distance(bbox)
            dist_to_stop=0
            #If between .75meters and 1 meter away, then send flag to stop
            if True or dist_to_stop > .75 and dist_to_stop < 1: #remove the OR TRUE statement once we worry about distace
                self.sign_in_range.publish(True)
                rospy.sleep(.1)

                drive_cmd = AckermannDriveStamped()
                drive_cmd.header.stamp = rospy.Time.now()
                drive_cmd.header.frame_id = "/base_link"
                drive_cmd.drive.steering_angle = 0 
                drive_cmd.drive.speed = 0
                self.drive_pub.publish(drive_cmd)

                rospy.sleep(1)

                self.sign_in_range.publish(False)
                #Sleeps for 5 seconds which should be enough time for the car to have driven
                #past the stop sign or atleast out of range to not consider it again
                self.time_to_wake = rospy.get_time() + 5

            else:
                self.sign_in_range.publish(False)

    #implement method that detects the distance between stop sign using some math, and publisihes true
    #when its in a certain range

    """
    Gets distance of object to camera
    """
    def get_distance(self, bbox):
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        #temporary until I figure out a way to do this
        return 0

    
    def state_callback(self, data):
        self.state_active = data.data

    

if __name__=="__main__":
    rospy.init_node('stop_sign_detecting')
    stop = StopSignControl()
    rospy.spin()
