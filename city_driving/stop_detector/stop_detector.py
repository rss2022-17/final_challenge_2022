import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from detector import StopSignDetector

from std_msgs.msg import Bool
class SignDetector:
    def __init__(self):
        self.detector = StopSignDetector()
        self.publisher = rospy.Publisher("/stop_sign_detected", Bool, queue_size = 10)
        self.subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.callback)

    def callback(self, img_msg):
        # Process image without CV Bridge
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        bgr_img = np_img[:,:,:-1]
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    
        stop_sign_detected = self.detector.predict(rgb_img)
        if stop_sign_detected:
            self.publisher.publish(True)
        


if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    rospy.spin()
