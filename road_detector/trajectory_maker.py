import numpy as np
import rospy
from homography_transformer import HomographyTransformer
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32, PoseArray
from utils import LineTrajectory

class RainbowTrajectory():
    """
    A class for applying line detection algorithms to the real robot. 
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: <INSERT TRAJECTORY TOPIC> (PoseArray) : the trajectory of the detected line in the world frame.
    
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """

    def __init__(self):
        self.trajectory = LineTrajectory("/planned_trajectory")
        self.traj_topic = "/trajectory/current"
        self.traj_pub = rospy.Publisher(self.traj_topic, PoseArray, queue_size=10)
        self.image_topic = "road_mask"

        self.homography = HomographyTransformer()

        #self.debug_pub = rospy.Publisher("/traj_debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        #self.prev_traj = Point32(0,0,0)

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
        image = np.array(image_msg).reshape((image_msg.width, image_msg.height))

        center = [ np.average(indices) for indices in np.where(image >= 255) ]
        x, y, = self.homography.transformUvToXy(center[0], center[1])
        new_point = Point32(x, y, 0)
        self.trajectory.addPoint(new_point)
        self.traj_pub.publish(self.trajectory.toPoseArray())
        # self.trajectory.publish_viz()
        #print("in image callback2!")

        # cv2.imwrite("debug.png", debug_img)
        #if (self.debug_pub.get_num_connections() > 0): self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

if __name__ == '__main__':
    try:
        rospy.init_node('RainbowTrajectory', anonymous=True)
        RainbowTrajectory()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
