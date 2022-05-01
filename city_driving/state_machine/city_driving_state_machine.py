import rospy

from std_msgs.msg import Bool

class StateMachine:

    def __init__(self):
        self.curState = None
        self.handlers = {}

    def add_state(self, state, handler):
        self.handlers[state.upper()] = handler

    def set_current_state(self, state):
        self.curState = state.upper()

    def run_event(self, event):
        if self.curState not in self.handlers.keys():
            return False
        newState = self.handlers[self.curState](event)
        # print self.curState + ' -> ' + newState
        self.curState = newState.upper()
        return True

class CityStateMachine:
    def __init__(self):
        self.fsm = StateMachine()
        self.stop_detect_topic = rospy.get_param("~stop_detected", "/stop_detected")
        self.car_wash_detect_topic = rospy.get_param("~car_wash_detected", "/car_wash_detected")
        self.line_follow_state_topic = rospy.get_param("~line_follower", "/line_follower")
        self.stop_state_topic = rospy.get_param("~stop_state", "/stop_state")
        self.car_wash_state_topic = rospy.get_param("~car_wash_state", "/car_wash_state")
        self.stop_sub = rospy.Subscriber(self.stop_detect_topic, Bool, self.stop_callback)
        self.wash_sub = rospy.Subscriber(self.car_wash_detect_topic, Bool, self.wash_callback)
        self.stopped_pub = rospy.Publisher(self.stop_state_topic, Bool, queue_size=1)
        self.washing_pub = rospy.Publisher(self.car_wash_state_topic, Bool, queue_size=1)
        self.following_pub = rospy.Publisher(self.line_follow_state_topic, Bool, queue_size=1)
        self.fsm.add_state('linefollower', self.line_follower_handler)
        self.fsm.add_state('stopdetector', self.stop_detection_handler)
        self.fsm.add_state('carwash', self.car_wash_handler)
        self.fsm.set_current_state('linefollower')
        msg_t = Bool()
        msg_t.data = True
        self.following_pub.publish(msg_t)

    def line_follower_handler(self, event):
        print "line follower"
        msg_f = Bool()
        msg_f.data = False
        msg_t = Bool()
        msg_t.data = True
        newState = 'linefollower'
        if event == 'stop_event':
            # stop detected state
            self.following_pub.publish(msg_f)
            rospy.sleep(0.01) # to ensure order of messages
            self.stopped_pub.publish(msg_t)
            newState = 'stopdetector'
        elif event == 'wash_event':
            # car wash detected state
            self.following_pub.publish(msg_f)
            rospy.sleep(0.01) # to ensure order of messages
            self.washing_pub.publish(msg_t)
            newState = 'carwash'
        # else:
        #     print "still line following :3"
        return newState

    def stop_detection_handler(self, event):
        # print "stop time"
        newState = 'stopdetector'
        if event == 'line_event':
            msg_t = Bool()
            msg_t.data = True
            self.following_pub.publish(msg_t)
            newState = 'linefollower'
        # else:
        #     print "stop still detected :O"
        return newState


    def car_wash_handler(self, event):
        # print "car wash time"
        newState = 'carwash'
        if event == 'line_event':
            msg_t = Bool()
            msg_t.data = True
            self.following_pub.publish(msg_t)
            newState = 'linefollower'
        # else:
        #     print "still in car wash :D"
        return newState

    def stop_callback(self, msg):
        if msg.data:
            self.fsm.run_event('stop_event')
        else:
            self.fsm.run_event('line_event')

    def wash_callback(self, msg):
        if msg.data:
            self.fsm.run_event('wash_event')
        else:
            self.fsm.run_event('line_event')


if __name__=="__main__":
    rospy.init_node('city_driving')
    csm = CityStateMachine()
    rospy.spin()

