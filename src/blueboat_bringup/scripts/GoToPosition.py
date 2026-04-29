#!/usr/bin/env python3
"""
GoToPosition Task
=================
Single-goal driver. Publishes one PoseStamped to /move_base_simple/goal,
records data, and exits.

Architectural note (matters with the updated guidance):
    Guidance now decides arrival on its own (map-frame distance to
    path end) and holds (0, psi) once arrived. This task's p3d-based
    arrival check is for *evaluation logging and exit timing only* —
    it does not influence the boat's behavior. Keeping p3d here gives
    you a ground-truth measurement independent of EKF/cartographer
    drift, which matters when comparing runs across configurations.

Records:
  1. Time-series CSV at RECORD_HZ.
  2. Per-goal artifacts (planned_path, planned_curvatures, waypoints).
  3. One-line summary at end (success, final position error, duration,
     path length) for batch comparison across runs.

rosrun blueboat_bringup GoToPosition.py --x 22.36 --y -3.70 --output run1.csv
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

from usv_msgs.msg import SpeedCourse


# ===========================================================================
# CONFIG
# ===========================================================================

DEFAULT_X      = 22.36
DEFAULT_Y      = -3.70
DEFAULT_YAW    = 0.0
DEFAULT_OUTPUT = "run1.csv"
TIMEOUT        = 60.0
RECORD_HZ      = 10.0

POSITION_TOL_M      = 0.8
GRACE_AFTER_ARRIVAL = 10.0

GOAL_FRAME     = "map"

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

TOPIC_GOAL          = "/move_base_simple/goal"

TOPIC_PLANNED_PATH       = "/planned_path"
TOPIC_PLANNED_CURVATURES = "/planned_curvatures"
TOPIC_WAYPOINTS          = "/waypoints"

# ===========================================================================

CSV_HEADER = [
    "wall_time", "ros_time",
    "odom_filt_x", "odom_filt_y", "odom_filt_z",
    "odom_filt_yaw", "odom_filt_vx", "odom_filt_vy",
    "odom_x", "odom_y", "odom_z",
    "odom_yaw", "odom_vx", "odom_vy",
    "p3d_x", "p3d_y", "p3d_z",
    "p3d_yaw", "p3d_vx", "p3d_vy",
    "left_thrust", "right_thrust",
    "cmd_speed", "cmd_course",
    "dbg_speed_actual", "dbg_speed_desired",
    "dbg_yaw_actual",   "dbg_yaw_desired",
    "dbg_thrust_left",  "dbg_thrust_right",
    "arrived",
]


def _stem(path):
    base, _ = os.path.splitext(path)
    return base


def _yaw_from_quat(q):
    _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


class GoToPositionTask:

    def __init__(self, x, y, yaw_deg, output_path, timeout,
                 position_tol, grace):
        self.x        = x
        self.y        = y
        self.yaw_rad  = math.radians(yaw_deg)
        self.output   = output_path
        self.timeout  = timeout
        self.pos_tol  = position_tol
        self.grace    = grace
        self.lock     = threading.Lock()

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

        self._arrived          = False
        self._arrival_ros_time = None

        # Path-length tracking (from p3d) for run summary
        self._prev_p3d_xy   = None
        self._path_length   = 0.0
        self._t_start       = None

        self._path_written       = False
        self._curvatures_written = False
        self._waypoints_written  = False

    @staticmethod
    def _unpack_odom(msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        return [p.x, p.y, p.z, yaw,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y]

    def cb_odom_filtered(self, msg):
        with self.lock: self._odom_filtered = self._unpack_odom(msg)

    def cb_odom(self, msg):
        with self.lock: self._odom = self._unpack_odom(msg)

    def cb_p3d(self, msg):
        # Ground truth from Gazebo. Used here for:
        #   - arrival latch (drives task termination)
        #   - path-length integration
        # Guidance does NOT use p3d; it has its own arrival logic.
        unpacked = self._unpack_odom(msg)
        with self.lock:
            self._p3d = unpacked
            px, py = unpacked[0], unpacked[1]

            if self._prev_p3d_xy is not None:
                dx = px - self._prev_p3d_xy[0]
                dy = py - self._prev_p3d_xy[1]
                self._path_length += math.hypot(dx, dy)
            self._prev_p3d_xy = (px, py)

            if not self._arrived:
                pos_err = math.hypot(px - self.x, py - self.y)
                if pos_err < self.pos_tol:
                    self._arrived          = True
                    self._arrival_ros_time = rospy.Time.now()
                    rospy.loginfo(
                        f"[GoToPosition] ARRIVED  pos_err={pos_err:.2f} m "
                        f"-> recording {self.grace}s grace "
                        f"(p3d-based eval; guidance independently holds)")

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
                rospy.loginfo(f"[GoToPosition] planned_path: {len(msg.poses)} poses -> {path_csv}")
                self._path_written = True
        except Exception as e:
            rospy.logwarn(f"[GoToPosition] failed to write planned_path: {e}")

    def cb_planned_curvatures(self, msg):
        curv_csv = f"{_stem(self.output)}_planned_curvatures.csv"
        try:
            with open(curv_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["i", "kappa"])
                for i, k in enumerate(msg.data):
                    w.writerow([i, k])
            if not self._curvatures_written:
                rospy.loginfo(f"[GoToPosition] planned_curvatures: {len(msg.data)} samples -> {curv_csv}")
                self._curvatures_written = True
        except Exception as e:
            rospy.logwarn(f"[GoToPosition] failed to write planned_curvatures: {e}")

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
                rospy.loginfo(f"[GoToPosition] waypoints: {len(msg.poses)} poses -> {wp_csv}")
                self._waypoints_written = True
        except Exception as e:
            rospy.logwarn(f"[GoToPosition] failed to write waypoints: {e}")

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

    def _write_summary(self):
        """One-line summary CSV next to the time-series for batch eval."""
        summary_csv = f"{_stem(self.output)}_summary.csv"
        with self.lock:
            success     = self._arrived
            final_x     = self._p3d[0]
            final_y     = self._p3d[1]
            duration    = ((self._arrival_ros_time - self._t_start).to_sec()
                           if success and self._t_start else None)
            path_len    = self._path_length
        if final_x is None or final_y is None:
            final_pos_err = None
        else:
            final_pos_err = math.hypot(final_x - self.x, final_y - self.y)
        try:
            with open(summary_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["success", "final_pos_err_m",
                            "duration_s", "path_length_m"])
                w.writerow([int(success),
                            f"{final_pos_err:.3f}" if final_pos_err is not None else "",
                            f"{duration:.3f}" if duration is not None else "",
                            f"{path_len:.3f}"])
            rospy.loginfo(
                f"[GoToPosition] summary: success={success}  "
                f"final_pos_err={final_pos_err}  duration={duration}  "
                f"path_len={path_len:.2f} m  -> {summary_csv}")
        except Exception as e:
            rospy.logwarn(f"[GoToPosition] failed to write summary: {e}")

    def run(self):
        rospy.init_node("go_to_position_task", anonymous=True)

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

        rospy.Subscriber(TOPIC_PLANNED_PATH,       Path,              self.cb_planned_path)
        rospy.Subscriber(TOPIC_PLANNED_CURVATURES, Float32MultiArray, self.cb_planned_curvatures)
        rospy.Subscriber(TOPIC_WAYPOINTS,          Path,              self.cb_waypoints)

        self._csvfile = open(self.output, "w", newline="")
        self._writer  = csv.writer(self._csvfile)
        self._writer.writerow(CSV_HEADER)
        rospy.loginfo(f"[GoToPosition] Recording -> {self.output}  @ {RECORD_HZ} Hz")

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
            f"[GoToPosition] Goal -> x={self.x}  y={self.y}  "
            f"yaw={math.degrees(self.yaw_rad):.1f} deg (yaw informational only)")
        rospy.loginfo(
            f"[GoToPosition] Eval tolerance (p3d): pos<{self.pos_tol} m. "
            f"Guidance handles its own arrival/hold independently.")
        rospy.loginfo(
            f"[GoToPosition] grace={self.grace}s, hard timeout={self.timeout}s")

        self._t_start = rospy.Time.now()
        rospy.Timer(rospy.Duration(1.0 / RECORD_HZ), self._write_row)

        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = (now - self._t_start).to_sec()

            if elapsed >= self.timeout:
                if self._arrived:
                    rospy.loginfo("[GoToPosition] hard timeout reached during grace")
                else:
                    rospy.logwarn(
                        f"[GoToPosition] hard timeout reached without arrival "
                        f"(t={elapsed:.1f}s)")
                break

            with self.lock:
                arrived   = self._arrived
                arrival_t = self._arrival_ros_time

            if arrived and arrival_t is not None:
                if (now - arrival_t).to_sec() >= self.grace:
                    rospy.loginfo(f"[GoToPosition] grace window complete (t={elapsed:.1f}s)")
                    break

            rate.sleep()

        self._csvfile.flush()
        self._csvfile.close()
        self._write_summary()
        rospy.loginfo(f"[GoToPosition] Done — saved {self.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--x",       type=float, default=DEFAULT_X)
    parser.add_argument("--y",       type=float, default=DEFAULT_Y)
    parser.add_argument("--yaw",     type=float, default=DEFAULT_YAW,    help="degrees (informational only)")
    parser.add_argument("--output",  type=str,   default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=TIMEOUT,        help="hard cap, seconds")
    parser.add_argument("--pos-tol", type=float, default=POSITION_TOL_M, help="meters")
    parser.add_argument("--grace",   type=float, default=GRACE_AFTER_ARRIVAL,
                        help="seconds to record after arrival")
    args = parser.parse_args()

    try:
        task = GoToPositionTask(args.x, args.y, args.yaw, args.output,
                                args.timeout, args.pos_tol, args.grace)
        task.run()
    except rospy.ROSInterruptException:
        pass