#!/usr/bin/env python3
# pwm_odom.py
#
# Build /odom from RC OUT PWMs of left/right thrusters.
# - Subscribes:  /mavros/rc/out   (mavros_msgs/RCOut)
# - Publishes:   /odom            (nav_msgs/Odometry)
#
# Logic:
#   1) Convert PWM -> ESC-domain command in [-1..+1] using your FWD/REV start thresholds.
#   2) Map ESC-domain to HULL-domain using LEFT_THRUST_DIR / RIGHT_THRUST_DIR (+1 or -1).
#   3) Differential-drive kinematics:
#         v = Vmax * 0.5 * (l_h + r_h)
#         w = Wmax * 0.5 * (r_h - l_h)
#   4) Integrate (x,y,yaw) and publish Odometry.
#
# If rc/out is missing or stale, publish zeros at RATE_HZ.

import math
import time
import rospy
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCOut
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Header

# ===================== CONFIG (match your main node) =====================

# Channels (ArduRover BlueBoat: SERVO1=right, SERVO3=left)
RIGHT_SERVO_NUM = 1  # MAIN1
LEFT_SERVO_NUM  = 3  # MAIN3

# PWM limits & neutral
PWM_MIN       = 1470
PWM_MAX       = 1540
NEUTRAL_LEFT  = 1500
NEUTRAL_RIGHT = 1500

# Start thresholds (first values that actually move your ESCs)
LEFT_FWD_START   = 1526
LEFT_REV_START   = 1486
RIGHT_FWD_START  = 1536   # your latest right fwd start
RIGHT_REV_START  = 1489

# Direction flips (+1 means hull-forward == ESC forward (>1500), -1 means hull-forward == ESC reverse (<1500))
LEFT_THRUST_DIR  = +1
RIGHT_THRUST_DIR = -1

# Odom tuning (boat-dependent; adjust!)
HULL_V_MAX_MPS = 1.00     # m/s when both sides are +1.0 (straight ahead)
HULL_W_MAX_RPS = 0.80     # rad/s when left=-1.0, right=+1.0 (pure spin)

# Deadband on normalized ESC command (ignore tiny noise)
THRUST_EPS = 0.02

# ROS + topic config
RATE_HZ        = 30
RC_TIMEOUT_S   = 0.5
ODOM_FRAME     = "odom"
BASE_LINK_FRAME= "base_link"
RC_OUT_TOPIC   = "/mavros/rc/out"

# ===================== HELPERS =====================

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def pwm_to_cmd(pwm, neutral, fwd_start, rev_start):
    """
    ESC-domain normalization (inverse of your cmd_to_pwm):
      > fwd_start  -> +[0..1]
      < rev_start  -> -[0..1]
      between      -> 0
    """
    if pwm is None:
        return 0.0
    # Forward region
    if pwm >= fwd_start:
        span = float(PWM_MAX - fwd_start) if PWM_MAX > fwd_start else 1.0
        c = (pwm - fwd_start) / span
        return clamp(c, 0.0, 1.0)
    # Reverse region
    if pwm <= rev_start:
        span = float(rev_start - PWM_MIN) if rev_start > PWM_MIN else 1.0
        c = - (rev_start - pwm) / span
        return clamp(c, -1.0, 0.0)
    # Dead zone near neutral
    return 0.0

def yaw_to_quat(z_yaw):
    """2D yaw -> quaternion (x=y=0)."""
    half = 0.5 * z_yaw
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q

# ===================== NODE =====================

class PwmOdomNode:
    def __init__(self):
        # State
        self.last_rc_time = 0.0
        self.right_pwm = None
        self.left_pwm = None

        # Integrated pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # ROS I/O
        self.pub_odom = rospy.Publisher("odom", Odometry, queue_size=10)
        rospy.Subscriber(RC_OUT_TOPIC, RCOut, self.cb_rcout, queue_size=10)

        self.rate = rospy.Rate(RATE_HZ)

    def ch_to_index(self, ch_num):
        """SERVO1 -> channels[0], SERVO3 -> channels[2]."""
        return int(ch_num) - 1

    def cb_rcout(self, msg: RCOut):
        # Guard array bounds
        ri = self.ch_to_index(RIGHT_SERVO_NUM)
        li = self.ch_to_index(LEFT_SERVO_NUM)
        if ri < len(msg.channels):
            self.right_pwm = int(msg.channels[ri])
        if li < len(msg.channels):
            self.left_pwm = int(msg.channels[li])
        self.last_rc_time = rospy.get_time()

    def compute_twist_from_pwm(self):
        """Return (vx, wz) in hull frame from current PWMs."""
        now = rospy.get_time()
        stale = (self.last_rc_time == 0.0) or ((now - self.last_rc_time) > RC_TIMEOUT_S)

        if stale or (self.right_pwm is None) or (self.left_pwm is None):
            # No RC data -> zeros
            return 0.0, 0.0

        # ESC-domain normalized thrust
        l_e = pwm_to_cmd(self.left_pwm,  NEUTRAL_LEFT,  LEFT_FWD_START,  LEFT_REV_START)
        r_e = pwm_to_cmd(self.right_pwm, NEUTRAL_RIGHT, RIGHT_FWD_START, RIGHT_REV_START)

        # Small deadband to ignore tiny whine/float
        if abs(l_e) < THRUST_EPS: l_e = 0.0
        if abs(r_e) < THRUST_EPS: r_e = 0.0

        # Map to HULL-domain (apply direction flips)
        l_h = l_e * (1 if LEFT_THRUST_DIR  >= 0 else -1)
        r_h = r_e * (1 if RIGHT_THRUST_DIR >= 0 else -1)

        # Differential kinematics -> velocities
        v  = HULL_V_MAX_MPS * 0.5 * (l_h + r_h)
        wz = HULL_W_MAX_RPS * 0.5 * (r_h - l_h)
        return v, wz

    def publish_odom(self, v, wz, dt):
        # Integrate pose (planar)
        self.yaw += wz * dt
        # normalize yaw for neatness
        if self.yaw > math.pi: 
            self.yaw -= 2.0*math.pi
        elif self.yaw < -math.pi:
            self.yaw += 2.0*math.pi

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt

        # Fill message
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ODOM_FRAME
        msg.child_frame_id = BASE_LINK_FRAME

        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = yaw_to_quat(self.yaw)

        msg.twist.twist.linear.x  = v
        msg.twist.twist.linear.y  = 0.0
        msg.twist.twist.linear.z  = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = wz

        self.pub_odom.publish(msg)

    def spin(self):
        last_t = rospy.get_time()
        while not rospy.is_shutdown():
            now = rospy.get_time()
            dt = max(1.0/RATE_HZ, now - last_t)
            last_t = now

            v, wz = self.compute_twist_from_pwm()
            self.publish_odom(v, wz, dt)
            self.rate.sleep()

def

