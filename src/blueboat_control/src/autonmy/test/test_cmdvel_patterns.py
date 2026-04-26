#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time

def make_twist(vx, wz):
    msg = Twist()
    msg.linear.x = vx   # forward/backward speed (m/s)
    msg.angular.z = wz  # yaw rate (rad/s)
    return msg

def main():
    rospy.init_node("cmdvel_test_patterns")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
    rospy.sleep(1.0)  # wait for pub setup

    rate = rospy.Rate(10)  # 10 Hz
    start_time = time.time()

    while not rospy.is_shutdown():
        t = time.time() - start_time

        # 0–5 s: straight back
        if 0 <= t < 5:
            cmd = make_twist(-0.3, 0.0)  # reverse at 0.5 m/s
            rospy.loginfo("Straight back")

        # 5–10 s: circle left (forward + left turn)
        elif 5 <= t < 10:
            cmd = make_twist(0.1, 0.3)
            rospy.loginfo("Circle left")

        # 10–15 s: circle right (forward + right turn)
        elif 10 <= t < 15:
            cmd = make_twist(0.1, -0.3)
            rospy.loginfo("Circle right")

        # 15–20 s: pure pivot left (no forward, just angular)
        elif 15 <= t < 20:
            cmd = make_twist(0.0, 0.3)  # just turning left
            rospy.loginfo("Pivot left (in place)")

        # 20–25 s: pure pivot right (no forward, just angular)
        elif 20 <= t < 25:
            cmd = make_twist(0.0, -0.3)  # just turning right
            rospy.loginfo("Pivot right (in place)")

        # After 25 s: stop neutral
        else:
            cmd = make_twist(0.0, 0.0)
            rospy.loginfo("Neutral (done)")

        pub.publish(cmd)
        rate.sleep()

if __name__ == "__main__":
    main()
