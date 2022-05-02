import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from detector import StopSignDetector
from std_msgs.msg import Bool

class StopSignControl():

    def __init__(self):
        self.bbox_sub = rospy.Subscriber("stop_sign_bbox", Float32MultiArray, self.bbox_callback)
        self.sign_in_range= rospy.Publisher("stop_detected", Bool, queue_size = 1)
        #self.image_sub = rospy.Subscriber("stop_sign_debug", Image, self.image_callback)
        
        #Currently just a guess for the transformation/distance for testing purposes


    """
    Callback method for bbox's
    """
    def bbox_callback(self, data):

        bbox = data.data
        #If no bounding box was found a.k.a no stop sign detected
        if bbox is None:
            self.sign_in_range.publish(False)
        
        else:
            dist_to_stop = self.get_distance(bbox)
            #If between .75meters and 1 meter away, then send flag to stop
            if dist_to_stop > .75 and dist_to_stop < 1 or True: #remove the OR TRUE statement once we worry about distace
                self.sign_in_range.publish(True)
                rospy.sleep(.1)
                self.sign_in_range.publish(False)
                #Sleeps for 5 seconds which should be enough time for the car to have driven
                #past the stop sign or atleast out of range to not consider it again
                rospy.sleep(5)

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

    

if __name__=="__main__":
    rospy.init_node('stop_sign_detecting')
    stop = StopSignControl()
    rospy.spin()
