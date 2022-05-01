#!/usr/bin/env python

import rospy
import numpy as np
import math

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)
        self.marker_pub = rospy.Publisher("/cone_marker", Marker, queue_size=2)

        self.line_following = rospy.get_param("~follow_lines", 0)

        self.use_velocity = 0.5 # m/s
        if self.line_following:
            self.parking_distance = 0.1 # meters; try playing with this number!
        else:
            self.parking_distance = 0.5 # m
        self.backup_fraction = 0.4
        self.relative_x = 0
        self.relative_y = 0
        

        #Added variables (different from starting code)
        self.park_distance_threshold = .075
        self.angle_threshold = 5 * np.pi/180 # 5 degrees but in radians
    
        self.kp = 0.5
        self.kd = 0

        #Other variables for PID controller
        self.last_error = 0
        # self.total_error = 0
        # self.max_error = 10
        self.last_derivative = 0
        self.alpha = 0.05
        self.last_time = rospy.Time.now()

        rospy.loginfo("Working - PD Controller")

    def relative_cone_callback(self, msg):
        # Cone coordinates relative to base link frame
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        
        self.drawMarker()

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
        #Check if the car is facing cone to an accurate extent. Additionally, check if it's the correct distance away; 
        #if it is, then make the velocity 0 so that the car is stopped facing the cone
        if self.facingCone(self.relative_x, self.relative_y):
            #rospy.loginfo("Is Facing Cone")
            activeAngle = 0.0
            if self.isCorrectDistanceAway(self.relative_x, self.relative_y):
                activeSpeed = 0.0
            elif distance_error < 0:
                activeSpeed = -self.use_velocity * self.backup_fraction

        else:
            if self.isCorrectDistanceAway(self.relative_x, self.relative_y):
                activeSpeed = -self.use_velocity * self.backup_fraction
                activeAngle = -activeAngle
            elif distance_error < 0:
                activeSpeed = -self.use_velocity * self.backup_fraction

        # if self.isCorrectDistanceAway(self.relative_x, self.relative_y):
        #         activeSpeed = 0.0

        #rospy.loginfo(activeAngle)
        #Setting Drive Commands for Publishing
        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = "base_link"
        drive_cmd.drive.steering_angle = activeAngle
        drive_cmd.drive.speed =  activeSpeed

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()
        #3 float values, x error, y error and distance error
        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x ** 2 + self.relative_y**2) - self.parking_distance

        #################################
        
        self.error_pub.publish(error_msg)
    
    def facingCone(self,cone_x, cone_y):
        angle = np.arctan2(cone_y, cone_x)
        return np.abs(angle) <= self.angle_threshold

        


    # Returns true if car is within self.parking_distance + threshold and self.parking_distance  - threshold
    def isCorrectDistanceAway(self, cone_x, cone_y):
        correctDistance = self.parking_distance
        currentDistance = np.sqrt(cone_x ** 2 + cone_y ** 2)
        return np.abs(correctDistance - currentDistance) <= self.park_distance_threshold

    def drawMarker(self):
        """
        Publish a marker to represent the cone in rviz
        """
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = .2
        marker.scale.y = .2
        marker.scale.z = .2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = .5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.relative_x
        marker.pose.position.y = self.relative_y
        self.marker_pub.publish(marker)


class PurePursuitController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)

        self.use_velocity = 0.5 # m/s
        self.parking_distance = 0.15 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        

        #Added variables (different from starting code)
        self.park_distance_threshold = .075
        self.angle_threshold = 5 * np.pi/180 # 7 degrees but in radians
        self.L = 0.32 #Distance between axles, might need to be remeasured 

        rospy.loginfo("Working PURE PURSUIT")

    def relative_cone_callback(self, msg):
        # Cone coordinates relative to base link frame
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()


        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd
        
        #Set Default Velocity
        activeSpeed = self.use_velocity
        
        angleToCone = np.arctan2(self.relative_y, self.relative_x)
        L1 = np.sqrt(self.relative_x ** 2 + self.relative_y**2) - self.parking_distance
        delta = np.arctan2(2*self.L*np.sin(angleToCone), L1) #steering angle

        if self.isCorrectDistanceAway(self.relative_x, self.relative_y):
            activeSpeed = 0.0

        drive_cmd.header.stamp = rospy.Time.now()
        drive_cmd.header.frame_id = "base_link"
        drive_cmd.drive.steering_angle = delta
        drive_cmd.drive.speed =  activeSpeed

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()
        #3 float values, x error, y error and distance error
        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)
        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x ** 2 + self.relative_y**2) - self.parking_distance

        #################################
        
        self.error_pub.publish(error_msg)
    

    # Returns true if car is within self.parking_distance + threshold and self.parking_distance  - threshold
    def isCorrectDistanceAway(self, cone_x, cone_y):
        correctDistance = self.parking_distance
        currentDistance = np.sqrt(cone_x ** 2 + cone_y ** 2)
        return np.abs(correctDistance - currentDistance) <= self.park_distance_threshold


if __name__ == '__main__':
    try:
        use_line_following = rospy.get_param("~follow_lines", 0)
        rospy.init_node('ParkingController', anonymous=True)

        if use_line_following:
            PurePursuitController()
        else:
            ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
