#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from usv_msgs.msg import SpeedCourse

current_yaw_rad = 0.0  # ENU radians, matches controller exactly

def odom_cb(msg):
    global current_yaw_rad
    q = msg.pose.pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    current_yaw_rad = math.atan2(siny, cosy)  # ENU radians, no conversion needed

def cmd_vel_cb(msg):
    sc = SpeedCourse()
    sc.speed = msg.linear.x
    
    # angular.z is already ENU rad/s — add small correction to current heading
    # No degree conversion, no compass conversion
    correction = msg.angular.z * 0.3  # tune this scale only
    sc.course = current_yaw_rad + correction
    
    # Normalize to (-pi, pi)
    while sc.course > math.pi:
        sc.course -= 2 * math.pi
    while sc.course < -math.pi:
        sc.course += 2 * math.pi
    
    pub.publish(sc)

rospy.init_node('cmd_vel_to_speed_heading')
rospy.Subscriber('/robot_localization/odometry/filtered', Odometry, odom_cb)
rospy.Subscriber('/cmd_vel', Twist, cmd_vel_cb)
pub = rospy.Publisher('/speed_heading', SpeedCourse, queue_size=1)
rospy.spin()
