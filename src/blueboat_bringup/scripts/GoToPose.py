#!/usr/bin/env python3
"""
GoToPose Task
=============
Publishes a goal to /move_base_simple/goal and records performance data.

Differs from GoToPosition:
  - Arrival is position AND yaw aligned (both within tolerance).
  - Arrival check uses /p3d_blueboat (ground truth) — not /odometry/filtered.
  - After arrival, records GRACE_AFTER_ARRIVAL seconds (default 10) and exits.
  - Outer --timeout still applies as a hard cap if arrival never happens.

Records two kinds of data:
  1. Time-series CSV at RECORD_HZ with all topics that matter for guidance
     and controller analysis. Adds an `arrived` flag column so you can mark
     which rows are inside the post-arrival grace window.
  2. Per-goal artifacts (planned_path, planned_curvatures, waypoints) saved
     once next to the main CSV.

rosrun blueboat_bringup GoToPose.py
rosrun blueboat_bringup GoToPose.py --x 22.36 --y -3.70 --yaw 90 --output run1.csv
"""

import rospy
import argparse
import csv
import math
import os
import threading
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Float32MultiArray
import tf.transformations as tft

# usv_msgs/SpeedCourse — the same message your guidance and controller use.
from usv_msgs.msg import SpeedCourse


# ===========================================================================
# GLOBAL CONFIG
# ===========================================================================

DEFAULT_X      = 22.36
DEFAULT_Y      = -3.70
DEFAULT_YAW    = 0.0          # degrees
DEFAULT_OUTPUT = "run1.csv"
TIMEOUT        = 60.0         # seconds — hard cap if arrival never happens
RECORD_HZ      = 10.0

# Arrival tolerances. Position is loose enough to be achievable in current/
# wind; yaw is reasonably tight so "aligned" actually means aligned. Bump
# yaw_tol up if the boat can't ever satisfy it (open-loop yaw after arrival).
POSITION_TOL_M     = 0.8      # meters
YAW_TOL_DEG        = 15.0     # degrees
GRACE_AFTER_ARRIVAL = 10.0    # seconds of recording after arrival latch

GOAL_FRAME     = "map"

# --- time-series subscribers ---
TOPIC_ODOM_FILTERED = "/odometry/filtered"
TOPIC_ODOM          = "/odom"
TOPIC_P3D           = "/p3d_blueboat"
TOPIC_LEFT_THRUST   = "/left_thrust_cmd"
TOPIC_RIGHT_THRUST  = "/right_thrust_cmd"
TOPIC_SPEED_HEADING = "/speed_heading"

TOPIC_DBG_SPEED_ACT = "/debug/speed_actual"
TOPIC_DBG_SPEED_DES = "/debug/speed_desired"
TOPIC_DBG_YAW_ACT   = "/debug/yaw_actual"
TOPIC_DBG_YAW_DES   = "/debug/yaw_desired"
TOPIC_DBG_THR_L     = "/debug/thrust_left"
TOPIC_DBG_THR_R     = "/debug/thrust_right"

# --- goal pub ---
TOPIC_GOAL          = "/move_base_simple/goal"

# --- per-goal artifact subscribers ---
TOPIC_PLANNED_PATH       = "/planned_path"
TOPIC_PLANNED_CURVATURES = "/planned_curvatures"
TOPIC_WAYPOINTS          = "/waypoints"

# ===========================================================================

CSV_HEADER = [
    "wall_time", "ros_time",
    # odom/filtered
    "odom_filt_x", "odom_filt_y", "odom_filt_z",
    "odom_filt_yaw", "odom_filt_vx", "odom_filt_vy",
    # odom
    "odom_x", "odom_y", "odom_z",
    "odom_yaw", "odom_vx", "odom_vy",
    # p3d (ground truth)
    "p3d_x", "p3d_y", "p3d_z",
    "p3d_yaw", "p3d_vx", "p3d_vy",
    # raw thrust commands
    "left_thrust", "right_thrust",
    # guidance -> controller
    "cmd_speed", "cmd_course",
    # controller debug
    "dbg_speed_actual", "dbg_speed_desired",
    "dbg_yaw_actual",   "dbg_yaw_desired",
    "dbg_thrust_left",  "dbg_thrust_right",
    # arrival flag (1 once latched and during grace, else 0)
    "arrived",
]


def _stem(path):
    base, _ = os.path.splitext(path)
    return base


def _yaw_from_quat(q):
    _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def _wrap_pi(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


class GoToPoseTask:

    def __init__(self, x, y, yaw_deg, output_path, timeout,
                 position_tol, yaw_tol_deg, grace):
        self.x        = x
        self.y        = y
        self.yaw_rad  = math.radians(yaw_deg)
        self.output   = output_path
        self.timeout  = timeout
        self.pos_tol  = position_tol
        self.yaw_tol  = math.radians(yaw_tol_deg)
        self.grace    = grace
        self.lock     = threading.Lock()

        # Latest cached values
        self._odom_filtered = [None] * 6
        self._odom          = [None] * 6
        self._p3d           = [None] * 6
        self._left_thrust   = None
        self._right_thrust  = None
        self._cmd_speed     = None
        self._cmd_course    = None
        self._dbg_speed_act = None
        self._dbg_speed_des = None
        self._dbg_yaw_act   = None
        self._dbg_yaw_des   = None
        self._dbg_thr_l     = None
        self._dbg_thr_r     = None

        # Arrival state
        self._arrived            = False
        self._arrival_ros_time   = None  # rospy.Time when latched

        # Once-per-goal artifact write tracking
        self._path_written       = False
        self._curvatures_written = False
        self._waypoints_written  = False

    # -----------------------------------------------------------------------
    # Time-series callbacks
    # -----------------------------------------------------------------------

    @staticmethod
    def _unpack_odom(msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        return [p.x, p.y, p.z, yaw,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y]

    def cb_odom_filtered(self, msg):
        with self.lock:
            self._odom_filtered = self._unpack_odom(msg)

    def cb_odom(self, msg):
        with self.lock:
            self._odom = self._unpack_odom(msg)

    def cb_p3d(self, msg):
        # p3d also drives the arrival check. Latch once; never un-latch.
        unpacked = self._unpack_odom(msg)
        with self.lock:
            self._p3d = unpacked
            if not self._arrived:
                px, py, _, pyaw, _, _ = unpacked
                pos_err = math.hypot(px - self.x, py - self.y)
                yaw_err = abs(_wrap_pi(pyaw - self.yaw_rad))
                if pos_err < self.pos_tol and yaw_err < self.yaw_tol:
                    self._arrived          = True
                    self._arrival_ros_time = rospy.Time.now()
                    rospy.loginfo(
                        f"[GoToPose] ARRIVED  pos_err={pos_err:.2f} m  "
                        f"yaw_err={math.degrees(yaw_err):.1f} deg  "
                        f"-> recording {self.grace}s grace")

    def cb_left_thrust(self, msg):
        with self.lock: self._left_thrust = msg.data
    def cb_right_thrust(self, msg):
        with self.lock: self._right_thrust = msg.data
    def cb_speed_heading(self, msg):
        with self.lock:
            self._cmd_speed  = msg.speed
            self._cmd_course = msg.course
    def cb_dbg_speed_act(self, msg):
        with self.lock: self._dbg_speed_act = msg.data
    def cb_dbg_speed_des(self, msg):
        with self.lock: self._dbg_speed_des = msg.data
    def cb_dbg_yaw_act(self, msg):
        with self.lock: self._dbg_yaw_act = msg.data
    def cb_dbg_yaw_des(self, msg):
        with self.lock: self._dbg_yaw_des = msg.data
    def cb_dbg_thr_l(self, msg):
        with self.lock: self._dbg_thr_l = msg.data
    def cb_dbg_thr_r(self, msg):
        with self.lock: self._dbg_thr_r = msg.data

    # -----------------------------------------------------------------------
    # Per-goal artifact callbacks
    # -----------------------------------------------------------------------

    def cb_planned_path(self, msg):
        path_csv = f"{_stem(self.output)}_planned_path.csv"
        try:
            with open(path_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["i", "x", "y", "yaw", "d"])
                for i, p in enumerate(msg.poses):
                    w.writerow([
                        i,
                        p.pose.position.x,
                        p.pose.position.y,
                        _yaw_from_quat(p.pose.orientation),
                        p.pose.position.z,
                    ])
            if not self._path_written:
                rospy.loginfo(f"[GoToPose] planned_path: {len(msg.poses)} poses -> {path_csv}")
                self._path_written = True
        except Exception as e:
            rospy.logwarn(f"[GoToPose] failed to write planned_path: {e}")

    def cb_planned_curvatures(self, msg):
        curv_csv = f"{_stem(self.output)}_planned_curvatures.csv"
        try:
            with open(curv_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["i", "kappa"])
                for i, k in enumerate(msg.data):
                    w.writerow([i, k])
            if not self._curvatures_written:
                rospy.loginfo(f"[GoToPose] planned_curvatures: {len(msg.data)} samples -> {curv_csv}")
                self._curvatures_written = True
        except Exception as e:
            rospy.logwarn(f"[GoToPose] failed to write planned_curvatures: {e}")

    def cb_waypoints(self, msg):
        wp_csv = f"{_stem(self.output)}_waypoints.csv"
        try:
            with open(wp_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["i", "x", "y", "yaw"])
                for i, p in enumerate(msg.poses):
                    w.writerow([
                        i,
                        p.pose.position.x,
                        p.pose.position.y,
                        _yaw_from_quat(p.pose.orientation),
                    ])
            if not self._waypoints_written:
                rospy.loginfo(f"[GoToPose] waypoints: {len(msg.poses)} poses -> {wp_csv}")
                self._waypoints_written = True
        except Exception as e:
            rospy.logwarn(f"[GoToPose] failed to write waypoints: {e}")

    # -----------------------------------------------------------------------
    # Timer callback
    # -----------------------------------------------------------------------

    def _write_row(self, event):
        with self.lock:
            row = (
                [time.time(), rospy.Time.now().to_sec()]
                + list(self._odom_filtered)
                + list(self._odom)
                + list(self._p3d)
                + [self._left_thrust, self._right_thrust]
                + [self._cmd_speed, self._cmd_course]
                + [self._dbg_speed_act, self._dbg_speed_des,
                   self._dbg_yaw_act,   self._dbg_yaw_des,
                   self._dbg_thr_l,     self._dbg_thr_r]
                + [1 if self._arrived else 0]
            )
        self._writer.writerow(row)

    # -----------------------------------------------------------------------

    def run(self):
        rospy.init_node("go_to_pose_task", anonymous=True)

        # Time-series subscribers
        rospy.Subscriber(TOPIC_ODOM_FILTERED, Odometry,    self.cb_odom_filtered)
        rospy.Subscriber(TOPIC_ODOM,          Odometry,    self.cb_odom)
        rospy.Subscriber(TOPIC_P3D,           Odometry,    self.cb_p3d)
        rospy.Subscriber(TOPIC_LEFT_THRUST,   Float32,     self.cb_left_thrust)
        rospy.Subscriber(TOPIC_RIGHT_THRUST,  Float32,     self.cb_right_thrust)
        rospy.Subscriber(TOPIC_SPEED_HEADING, SpeedCourse, self.cb_speed_heading)

        rospy.Subscriber(TOPIC_DBG_SPEED_ACT, Float32, self.cb_dbg_speed_act)
        rospy.Subscriber(TOPIC_DBG_SPEED_DES, Float32, self.cb_dbg_speed_des)
        rospy.Subscriber(TOPIC_DBG_YAW_ACT,   Float32, self.cb_dbg_yaw_act)
        rospy.Subscriber(TOPIC_DBG_YAW_DES,   Float32, self.cb_dbg_yaw_des)
        rospy.Subscriber(TOPIC_DBG_THR_L,     Float32, self.cb_dbg_thr_l)
        rospy.Subscriber(TOPIC_DBG_THR_R,     Float32, self.cb_dbg_thr_r)

        # Per-goal artifact subscribers
        rospy.Subscriber(TOPIC_PLANNED_PATH,       Path,              self.cb_planned_path)
        rospy.Subscriber(TOPIC_PLANNED_CURVATURES, Float32MultiArray, self.cb_planned_curvatures)
        rospy.Subscriber(TOPIC_WAYPOINTS,          Path,              self.cb_waypoints)

        # CSV
        self._csvfile = open(self.output, "w", newline="")
        self._writer  = csv.writer(self._csvfile)
        self._writer.writerow(CSV_HEADER)
        rospy.loginfo(f"[GoToPose] Recording -> {self.output}  @ {RECORD_HZ} Hz")

        # Goal
        goal_pub = rospy.Publisher(TOPIC_GOAL, PoseStamped, queue_size=1, latch=True)
        rospy.sleep(1.0)

        goal = PoseStamped()
        goal.header.frame_id = GOAL_FRAME
        goal.header.stamp    = rospy.Time.now()
        goal.pose.position.x = self.x
        goal.pose.position.y = self.y
        goal.pose.position.z = 0.0
        q = tft.quaternion_from_euler(0, 0, self.yaw_rad)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        goal_pub.publish(goal)
        rospy.loginfo(
            f"[GoToPose] Goal -> x={self.x}  y={self.y}  "
            f"yaw={math.degrees(self.yaw_rad):.1f} deg")
        rospy.loginfo(
            f"[GoToPose] Tolerances: pos<{self.pos_tol} m, "
            f"yaw<{math.degrees(self.yaw_tol):.1f} deg, "
            f"grace={self.grace}s, hard timeout={self.timeout}s")

        rospy.Timer(rospy.Duration(1.0 / RECORD_HZ), self._write_row)

        # Main loop: poll for arrival + grace, with hard timeout.
        # Uses ROS time so it respects sim clock.
        start = rospy.Time.now()
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = (now - start).to_sec()

            if elapsed >= self.timeout:
                if self._arrived:
                    rospy.loginfo("[GoToPose] hard timeout reached during grace")
                else:
                    rospy.logwarn(
                        f"[GoToPose] hard timeout reached without arrival "
                        f"(t={elapsed:.1f}s)")
                break

            with self.lock:
                arrived = self._arrived
                arrival_t = self._arrival_ros_time

            if arrived and arrival_t is not None:
                if (now - arrival_t).to_sec() >= self.grace:
                    rospy.loginfo(
                        f"[GoToPose] grace window complete "
                        f"(t={elapsed:.1f}s)")
                    break

            rate.sleep()

        self._csvfile.flush()
        self._csvfile.close()
        rospy.loginfo(f"[GoToPose] Done — saved {self.output}")


# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--x",       type=float, default=DEFAULT_X)
    parser.add_argument("--y",       type=float, default=DEFAULT_Y)
    parser.add_argument("--yaw",     type=float, default=DEFAULT_YAW,    help="degrees")
    parser.add_argument("--output",  type=str,   default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=TIMEOUT,        help="hard cap, seconds")
    parser.add_argument("--pos-tol", type=float, default=POSITION_TOL_M, help="meters")
    parser.add_argument("--yaw-tol", type=float, default=YAW_TOL_DEG,    help="degrees")
    parser.add_argument("--grace",   type=float, default=GRACE_AFTER_ARRIVAL,
                        help="seconds to record after arrival")
    args = parser.parse_args()

    try:
        task = GoToPoseTask(args.x, args.y, args.yaw, args.output,
                            args.timeout, args.pos_tol, args.yaw_tol, args.grace)
        task.run()
    except rospy.ROSInterruptException:
        pass