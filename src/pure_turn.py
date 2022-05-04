#!/usr/bin/env python

import rospy
import numpy as np
import time
import utils
import tf

from geometry_msgs.msg import PoseArray, PoseStamped, Point, PointStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64 import Bool

class PureTurn(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic")
        self.drive_topic      = rospy.get_param("~turn_topic", "/vesc/high_level/ackermann_cmd_mux/input/nav_2")
        self.speed            = float(rospy.get_param("~speed", 0.5))
        #self.wrap             = # FILL IN #
        self.wheelbase_length = 0.32#
        self.turn_sub = rospy.Subscriber("/turn_state/", Bool, self.turn_callback, queue_size=1)
        self.direction_sub = rospy.Subscriber("/turn_direction/", Bool, self.direction_callback, queue_size=1)
        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        self.angle = 0

    def turn_callback(self, msg):
        ''' Clears the currently followed trajectory, and loads the new one from the message
        '''
        if msg.data:
            drive_cmd = AckermannDriveStamped()
            drive_cmd.header.stamp = rospy.Time.now()
            drive_cmd.header.frame_id = "/base_link"
            drive_cmd.drive.steering_angle = self.angle
            drive_cmd.drive.speed = self.speed
            self.drive_pub.publish(drive_cmd)

    def direction_callback(self, msg):
        if msg.data:
            self.angle = 1
        else:
            self.angle = -1

if __name__=="__main__":
    rospy.init_node("pure_turn")
    pf = PureTurn()
    rospy.spin()
