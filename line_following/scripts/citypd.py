#!/usr/bin/env python

import rospy
import numpy as np
import math

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point32

#Copied from visual servoing lab, meant for following line in city
class CityController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/linecenter", Point32,
            self.relative_point_callback)

        self.drive_pub = rospy.Publisher("/vesc/high_level/ackermann_cmd_mux/input/nav_3",
            AckermannDriveStamped, queue_size=10)
        self.use_velocity = 0.5 # m/s
        self.parking_distance = 0.1 # meters; try playing with this number!
        self.backup_fraction = 0.4
        self.relative_x = 0
        self.relative_y = 0
        

        #Added variables (different from starting code)
        self.park_distance_threshold = .075
        self.angle_threshold = 5 * np.pi/180 # 5 degrees but in radians
    
        self.kp = 1
        self.kd = 0

        #Other variables for PD controller
        self.last_error = 0
        # self.total_error = 0
        # self.max_error = 10
        self.last_derivative = 0
        self.alpha = 0.05
        self.last_time = rospy.Time.now()

        rospy.loginfo("Working - PD Controller")

    def relative_point_callback(self, msg):
        # Cone coordinates relative to base link frame
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        

        drive_cmd = AckermannDriveStamped()


        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd
        
        #Set Default Velocity
        activeSpeed = self.use_velocity
        activeAngle = 0
        
        #Start PID controller
        currTime = rospy.Time.now()
        angleToCone = np.arctan2(self.relative_y, self.relative_x)
        error = angleToCone

        dt = (currTime.to_sec() - self.last_time.to_sec())
        de = self.kd * (error - self.last_error)/dt

        P = self.kp*error
        D = self.alpha*de + (1-self.alpha)*self.last_derivative
        
        activeAngle = P + D
        self.last_derivative = D
        self.last_error = error 
        self.last_time = currTime

        #End PID controller


        distance_error = np.sqrt(self.relative_x ** 2 + self.relative_y**2) - self.parking_distance
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = "base_link"
        drive_cmd.drive.steering_angle = activeAngle
        drive_cmd.drive.speed =  activeSpeed

        #################################

        self.drive_pub.publish(drive_cmd)


if __name__ == '__main__':
    try:
        rospy.init_node('CityController', anonymous=True)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
