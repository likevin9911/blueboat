#!/usr/bin/env python3
import time, sys, signal, atexit
import rospy
from geometry_msgs.msg import Twist
from pymavlink import mavutil

# ===================== CONSTANTS (edit here) =====================

# ROS
RATE_HZ       = 30
CMD_TIMEOUT_S = 0.5

# Motion scaling
MAX_LINEAR   = 1.0   # m/s -> full forward/reverse
MAX_ANGULAR  = 1.0   # rad/s -> full turn
LINEAR_GAIN  = 0.25
ANGULAR_GAIN = 0.00

# Deadbands & pivot thresholds
LINEAR_DEADBAND      = 0.02
ANGULAR_DEADBAND     = 0.02
TURN_ONLY_W_THRESH   = 0.15   # |w_n| above -> pivot mode (if |v_n| small)
LINEAR_LOCKOUT_TURN  = 0.10   # |v_n| must be below this to allow pivot

# PWM ranges
NEUTRAL_PWM = 1500
MIN_PWM     = 1470
MAX_PWM     = 1540

# Motor-specific start thresholds (skip dead zone)
LEFT_FWD_START  = 1526
LEFT_REV_START  = 1489
RIGHT_FWD_START = 1531
RIGHT_REV_START = 1491

# MAVLink connection
CONN_STR    = "udpin:0.0.0.0:14550"
SOURCE_SYS  = 1
SOURCE_COMP = 190
TARGET_SYS  = 1
TARGET_COMP = 1

# Channel mapping (ArduPilot convention)
CH_RIGHT = 1   # SERVO1_FUNCTION
CH_LEFT  = 3   # SERVO3_FUNCTION

# SERVO function setup: 1 = RCPASS, 51 = RCIN1, 53 = RCIN3
# Use RCIN when you want to feed raw RC inputs into specific servos.
MODE_RIGHT = 2  # 1=RCPASS, 2=RCIN
MODE_LEFT  = 2

# Stop sequence “forward blip” (helps escape any reverse latch)
BLIP_DELTA   = 28      # microseconds above neutral
BLIP_TIME_S  = 0.30

# ========================= MAVLINK HELPER =========================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class MavRC:
    def __init__(self):
        self.m = mavutil.mavlink_connection(
            CONN_STR, source_system=SOURCE_SYS, source_component=SOURCE_COMP, dialect="ardupilotmega"
        )
        hb = self.m.wait_heartbeat(timeout=10.0)
        if not hb:
            raise RuntimeError("No MAVLink HEARTBEAT received")
        self.m.target_system = TARGET_SYS
        self.m.target_component = TARGET_COMP

    def set_param(self, name, value, ptype=mavutil.mavlink.MAV_PARAM_TYPE_REAL32):
        self.m.mav.param_set_send(
            self.m.target_system, self.m.target_component,
            name.encode('ascii'), float(value), ptype
        )

    def set_servo_functions(self, mode_right, mode_left):
        func = {"RCPASS": 1, "RCIN1": 51, "RCIN3": 53}
        fr = func["RCPASS"] if mode_right == 1 else func["RCIN1"]
        fl = func["RCPASS"] if mode_left  == 1 else func["RCIN3"]
        self.set_param("SERVO1_FUNCTION", fr)
        self.set_param("SERVO3_FUNCTION", fl)

    def set_mode_manual(self):
        self.m.mav.set_mode_send(
            TARGET_SYS,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            0
        )

    def arm(self, val=True):
        self.m.mav.command_long_send(
            TARGET_SYS, TARGET_COMP,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1.0 if val else 0.0, 0, 0, 0, 0, 0, 0
        )

    def rc_override(self, ch_right=None, ch_left=None):
        # ArduPilot expects 8 values, 65535 = ignore channel
        vals = [65535] * 8
        if ch_right is not None:
            vals[CH_RIGHT - 1] = int(ch_right)
        if ch_left is not None:
            vals[CH_LEFT - 1] = int(ch_left)
        self.m.mav.rc_channels_override_send(
            self.m.target_system, self.m.target_component, *vals
        )

    def release_override(self, repeats=5):
        for _ in range(repeats):
            self.m.mav.rc_channels_override_send(
                self.m.target_system, self.m.target_component,
                65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535
            )
            time.sleep(0.05)

    def neutral_spam(self, secs=0.5):
        t_end = time.time() + secs
        while time.time() < t_end:
            self.rc_override(NEUTRAL_PWM, NEUTRAL_PWM)
            time.sleep(0.05)

    def stop_sequence(self, with_blip=True):
        try:
            if with_blip:
                blip = NEUTRAL_PWM + BLIP_DELTA
                self.rc_override(blip, blip)
                time.sleep(BLIP_TIME_S)
            self.neutral_spam(0.5)
            self.release_override()
            self.arm(False)
        except Exception:
            pass

# ========================= CONVERSION LOGIC =========================

def apply_deadband(x, db):
    return 0.0 if abs(x) < db else x

def clip1(x):
    return clamp(x, -1.0, 1.0)

def cmd_to_pwm(cmd, fwd_start, rev_start):
    """
    cmd in [-1, 1]
      >0 => map to [fwd_start .. MAX_PWM]
       0 => NEUTRAL_PWM
      <0 => map to [MIN_PWM .. rev_start]
    """
    if cmd > 0.0:
        return int(round(fwd_start + cmd * (MAX_PWM - fwd_start)))
    elif cmd < 0.0:
        mag = abs(cmd)
        return int(round(rev_start - mag * (rev_start - MIN_PWM)))
    else:
        return NEUTRAL_PWM

# ========================= ROS NODE =========================

class CmdVelToPWMNode:
    def __init__(self):
        self.mav = MavRC()
        self.mav.set_servo_functions(MODE_RIGHT, MODE_LEFT)
        self.mav.set_mode_manual()
        self.mav.arm(True)

        self.last_rx = time.time()
        rospy.Subscriber("cmd_vel", Twist, self.cb_cmdvel)

        # Clean shutdown: neutral + forward blip + disarm
        def _cleanup(*_):
            self.mav.stop_sequence(with_blip=True)
            try:
                self.mav.m.close()
            except Exception:
                pass
            sys.exit(0)

        atexit.register(_cleanup)
        signal.signal(signal.SIGINT, _cleanup)
        signal.signal(signal.SIGTERM, _cleanup)

    def cb_cmdvel(self, msg: Twist):
        self.last_rx = time.time()
        v = msg.linear.x
        w = msg.angular.z

        # Normalize and deadband
        v_n = clip1(apply_deadband(v / MAX_LINEAR,  LINEAR_DEADBAND))
        w_n = clip1(apply_deadband(w / MAX_ANGULAR, ANGULAR_DEADBAND))

        if LINEAR_GAIN != 0.0:
            v_n = clip1(v_n * (1.0 + LINEAR_GAIN))
        if ANGULAR_GAIN != 0.0:
            w_n = clip1(w_n * (1.0 + ANGULAR_GAIN))

        # Decide control mode
        if abs(w_n) > TURN_ONLY_W_THRESH and abs(v_n) < LINEAR_LOCKOUT_TURN:
            # Pivot: counter-rotating
            strength = clamp(abs(w_n), 0.0, 1.0)
            if w_n > 0:   # left turn => right motor forward
                left_cmd, right_cmd = -strength, strength
            else:         # right turn => left motor forward
                left_cmd, right_cmd = strength, -strength
        else:
            # Arcade mix
            left_cmd  = clip1(v_n - w_n)
            right_cmd = clip1(v_n + w_n)

        # Convert to PWM (note: RIGHT is channel 1, LEFT is channel 3)
        left_pwm  = cmd_to_pwm(left_cmd,  LEFT_FWD_START,  LEFT_REV_START)
        right_pwm = cmd_to_pwm(right_cmd, RIGHT_FWD_START, RIGHT_REV_START)

        # Clamp to safety bounds and send
        left_pwm  = clamp(left_pwm,  MIN_PWM, MAX_PWM)
        right_pwm = clamp(right_pwm, MIN_PWM, MAX_PWM)
        self.mav.rc_override(ch_right=right_pwm, ch_left=left_pwm)

    def spin(self):
        rate = rospy.Rate(RATE_HZ)
        while not rospy.is_shutdown():
            # Safety neutral if no recent cmd_vel
            if time.time() - self.last_rx > CMD_TIMEOUT_S:
                self.mav.rc_override(ch_right=NEUTRAL_PWM, ch_left=NEUTRAL_PWM)
            rate.sleep()

def main():
    rospy.init_node("cmdvel_to_pwm_mav")
    node = CmdVelToPWMNode()
    node.spin()

if __name__ == "__main__":
    main()
