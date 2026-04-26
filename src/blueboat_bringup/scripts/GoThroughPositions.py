#!/usr/bin/env python3
"""
GoThroughPositions Task
=======================
Drives the boat through a sequence of waypoints, publishing each as a
new goal on /move_base_simple/goal once the previous one is reached.

Arrival rule:
  - Intermediate waypoints: position only, within --pos-tol meters.
  - Final waypoint:         position AND yaw, within --pos-tol meters
                            and --yaw-tol degrees.

Why the asymmetry: holding yaw at every intermediate stalls the run on
benchmarks where the boat just needs to pass through. The final pose is
where alignment actually matters for scoring "did we reach the pose."

After the final waypoint's arrival latch, records --grace seconds and
exits. Outer --timeout still applies as a hard cap.

Records two kinds of data:
  1. Time-series CSV at RECORD_HZ. A `wp_index` column tracks which
     waypoint is currently active (0-based), and `arrived_final` flags
     rows inside the post-final-arrival grace window.
  2. Per-goal artifacts (planned_path, planned_curvatures, waypoints)
     saved as <stem>_wp{i}_<artifact>.csv — one set per waypoint segment.

Examples:
  rosrun blueboat_bringup GoThroughPositions.py \\
      --waypoints "10,0,0  20,5,90  22.36,-3.7,0" \\
      --output multi.csv

  # CSV-style waypoint string also works:
  rosrun blueboat_bringup GoThroughPositions.py \\
      --waypoints "10,0,0;20,5,90;22.36,-3.7,0"

Waypoint format: "x,y,yaw_deg" tuples separated by spaces or semicolons.
The last tuple is treated as the final waypoint.
"""

import rospy
import argparse
import csv
import math
import os
import re
import threading
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Float32MultiArray
import tf.transformations as tft

from usv_msgs.msg import SpeedCourse


# ===========================================================================
# GLOBAL CONFIG
# ===========================================================================

DEFAULT_WAYPOINTS = "10,0,0  20,5,90  22.36,-3.7,0"
DEFAULT_OUTPUT    = "multi.csv"
TIMEOUT           = 180.0     # hard cap for the whole sequence
RECORD_HZ         = 10.0

POSITION_TOL_M       = 0.8
YAW_TOL_DEG          = 15.0
GRACE_AFTER_ARRIVAL  = 10.0   # only applied after the FINAL waypoint

GOAL_FRAME           = "map"

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
    # Sequence state
    "wp_index",        # which waypoint is currently active (0-based)
    "arrived_final",   # 1 once the final waypoint has been reached
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


def parse_waypoints(s):
    """
    Accept "x,y,yaw  x,y,yaw" or "x,y,yaw;x,y,yaw" — splits on whitespace
    OR semicolons. yaw is in degrees.
    Returns list of (x, y, yaw_rad).
    """
    if not s or not s.strip():
        raise ValueError("empty --waypoints string")
    # Split on any run of whitespace or semicolons
    chunks = [c for c in re.split(r"[\s;]+", s.strip()) if c]
    out = []
    for i, chunk in enumerate(chunks):
        parts = chunk.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"waypoint #{i+1} '{chunk}' must be 'x,y,yaw_deg'")
        try:
            x = float(parts[0]); y = float(parts[1]); yaw = float(parts[2])
        except ValueError:
            raise ValueError(f"waypoint #{i+1} '{chunk}' has non-numeric fields")
        out.append((x, y, math.radians(yaw)))
    if not out:
        raise ValueError("no waypoints parsed")
    return out


class GoThroughPositionsTask:

    def __init__(self, waypoints, output_path, timeout,
                 position_tol, yaw_tol_deg, grace):
        self.waypoints = waypoints           # list of (x, y, yaw_rad)
        self.output    = output_path
        self.timeout   = timeout
        self.pos_tol   = position_tol
        self.yaw_tol   = math.radians(yaw_tol_deg)
        self.grace     = grace
        self.lock      = threading.Lock()

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

        # Sequence state
        self._wp_index           = 0       # currently active waypoint
        self._final_arrived      = False
        self._final_arrival_time = None
        # Per-waypoint arrival flag, used to drive advance from cb_p3d
        self._current_wp_arrived = False
        # Goal publisher set in run(); needed from cb_p3d to publish next goal
        self._goal_pub           = None

        # Per-segment artifact tracking — written once per waypoint index.
        self._artifact_seen = {
            "path":        set(),  # wp_indices we've already written path for
            "curvatures":  set(),
            "waypoints":   set(),
        }

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
        with self.lock: self._odom_filtered = self._unpack_odom(msg)

    def cb_odom(self, msg):
        with self.lock: self._odom = self._unpack_odom(msg)

    def cb_p3d(self, msg):
        # p3d drives both pose logging and per-waypoint arrival.
        unpacked = self._unpack_odom(msg)
        advance_to = None
        publish_goal_for = None
        with self.lock:
            self._p3d = unpacked
            if self._final_arrived:
                # Already done — just keep recording.
                return

            idx = self._wp_index
            if idx >= len(self.waypoints):
                return  # shouldn't happen, defensive

            wx, wy, wyaw = self.waypoints[idx]
            px, py, _, pyaw, _, _ = unpacked
            pos_err = math.hypot(px - wx, py - wy)
            yaw_err = abs(_wrap_pi(pyaw - wyaw))

            is_final = (idx == len(self.waypoints) - 1)
            if is_final:
                ok = (pos_err < self.pos_tol) and (yaw_err < self.yaw_tol)
            else:
                ok = (pos_err < self.pos_tol)

            if ok and not self._current_wp_arrived:
                self._current_wp_arrived = True
                if is_final:
                    self._final_arrived      = True
                    self._final_arrival_time = rospy.Time.now()
                    rospy.loginfo(
                        f"[GoThrough] FINAL ARRIVED  "
                        f"pos_err={pos_err:.2f} m  "
                        f"yaw_err={math.degrees(yaw_err):.1f} deg  "
                        f"-> recording {self.grace}s grace")
                else:
                    rospy.loginfo(
                        f"[GoThrough] wp[{idx}] reached  "
                        f"pos_err={pos_err:.2f} m  -> advancing")
                    advance_to               = idx + 1
                    publish_goal_for         = idx + 1
                    self._wp_index           = advance_to
                    self._current_wp_arrived = False

        # Publish next goal OUTSIDE the lock. ROS publish is generally safe
        # from any thread but holding the lock across it is unnecessary.
        if publish_goal_for is not None:
            self._publish_goal(publish_goal_for)

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
    # Per-segment artifact callbacks. Each one writes a CSV per waypoint
    # index, so a 3-waypoint run produces _wp0_planned_path.csv,
    # _wp1_planned_path.csv, _wp2_planned_path.csv.
    # -----------------------------------------------------------------------

    def _current_wp_index(self):
        with self.lock:
            return self._wp_index

    def cb_planned_path(self, msg):
        idx = self._current_wp_index()
        if idx in self._artifact_seen["path"]:
            return  # already wrote one for this waypoint
        path_csv = f"{_stem(self.output)}_wp{idx}_planned_path.csv"
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
            rospy.loginfo(
                f"[GoThrough] wp[{idx}] planned_path: "
                f"{len(msg.poses)} poses -> {path_csv}")
            self._artifact_seen["path"].add(idx)
        except Exception as e:
            rospy.logwarn(f"[GoThrough] failed to write planned_path: {e}")

    def cb_planned_curvatures(self, msg):
        idx = self._current_wp_index()
        if idx in self._artifact_seen["curvatures"]:
            return
        curv_csv = f"{_stem(self.output)}_wp{idx}_planned_curvatures.csv"
        try:
            with open(curv_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["i", "kappa"])
                for i, k in enumerate(msg.data):
                    w.writerow([i, k])
            rospy.loginfo(
                f"[GoThrough] wp[{idx}] planned_curvatures: "
                f"{len(msg.data)} samples -> {curv_csv}")
            self._artifact_seen["curvatures"].add(idx)
        except Exception as e:
            rospy.logwarn(f"[GoThrough] failed to write planned_curvatures: {e}")

    def cb_waypoints(self, msg):
        idx = self._current_wp_index()
        if idx in self._artifact_seen["waypoints"]:
            return
        wp_csv = f"{_stem(self.output)}_wp{idx}_waypoints.csv"
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
            rospy.loginfo(
                f"[GoThrough] wp[{idx}] waypoints: "
                f"{len(msg.poses)} poses -> {wp_csv}")
            self._artifact_seen["waypoints"].add(idx)
        except Exception as e:
            rospy.logwarn(f"[GoThrough] failed to write waypoints: {e}")

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
                + [self._wp_index, 1 if self._final_arrived else 0]
            )
        self._writer.writerow(row)

    # -----------------------------------------------------------------------
    # Goal publishing
    # -----------------------------------------------------------------------

    def _publish_goal(self, idx):
        if self._goal_pub is None:
            return
        if idx >= len(self.waypoints):
            return
        x, y, yaw_rad = self.waypoints[idx]
        goal = PoseStamped()
        goal.header.frame_id = GOAL_FRAME
        goal.header.stamp    = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        q = tft.quaternion_from_euler(0, 0, yaw_rad)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]
        self._goal_pub.publish(goal)
        is_final = (idx == len(self.waypoints) - 1)
        rospy.loginfo(
            f"[GoThrough] goal[{idx}{'/final' if is_final else ''}] "
            f"-> x={x}  y={y}  yaw={math.degrees(yaw_rad):.1f} deg")

    # -----------------------------------------------------------------------

    def run(self):
        rospy.init_node("go_through_positions_task", anonymous=True)

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
        rospy.loginfo(f"[GoThrough] Recording -> {self.output}  @ {RECORD_HZ} Hz")

        self._goal_pub = rospy.Publisher(TOPIC_GOAL, PoseStamped,
                                         queue_size=1, latch=True)
        rospy.sleep(1.0)

        rospy.loginfo(f"[GoThrough] {len(self.waypoints)} waypoints:")
        for i, (x, y, yaw_rad) in enumerate(self.waypoints):
            tag = " (FINAL)" if i == len(self.waypoints) - 1 else ""
            rospy.loginfo(
                f"  [{i}] x={x}  y={y}  yaw={math.degrees(yaw_rad):.1f} deg{tag}")
        rospy.loginfo(
            f"[GoThrough] Tolerances: pos<{self.pos_tol} m, "
            f"yaw<{math.degrees(self.yaw_tol):.1f} deg (final only), "
            f"grace={self.grace}s, hard timeout={self.timeout}s")

        # Publish first goal
        self._publish_goal(0)

        rospy.Timer(rospy.Duration(1.0 / RECORD_HZ), self._write_row)

        # Main loop with hard cap and post-final grace.
        start = rospy.Time.now()
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = (now - start).to_sec()

            if elapsed >= self.timeout:
                with self.lock:
                    final = self._final_arrived
                    idx = self._wp_index
                if final:
                    rospy.loginfo("[GoThrough] hard timeout reached during grace")
                else:
                    rospy.logwarn(
                        f"[GoThrough] hard timeout reached at wp[{idx}]/"
                        f"{len(self.waypoints)} (no final arrival)")
                break

            with self.lock:
                final     = self._final_arrived
                arrival_t = self._final_arrival_time

            if final and arrival_t is not None:
                if (now - arrival_t).to_sec() >= self.grace:
                    rospy.loginfo(f"[GoThrough] grace window complete (t={elapsed:.1f}s)")
                    break

            rate.sleep()

        self._csvfile.flush()
        self._csvfile.close()
        rospy.loginfo(f"[GoThrough] Done — saved {self.output}")


# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("Drive through a sequence of waypoints. "
                     "Intermediates: position-only arrival. Final: position+yaw."),
    )
    parser.add_argument("--waypoints", type=str, default=DEFAULT_WAYPOINTS,
                        help="space- or semicolon-separated 'x,y,yaw_deg' tuples")
    parser.add_argument("--output",    type=str,   default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout",   type=float, default=TIMEOUT,
                        help="hard cap for whole sequence, seconds")
    parser.add_argument("--pos-tol",   type=float, default=POSITION_TOL_M, help="meters")
    parser.add_argument("--yaw-tol",   type=float, default=YAW_TOL_DEG,
                        help="degrees, applied at FINAL waypoint only")
    parser.add_argument("--grace",     type=float, default=GRACE_AFTER_ARRIVAL,
                        help="seconds to record after final arrival")
    args = parser.parse_args()

    try:
        wps = parse_waypoints(args.waypoints)
    except ValueError as e:
        print(f"ERROR parsing --waypoints: {e}")
        raise SystemExit(2)

    try:
        task = GoThroughPositionsTask(
            wps, args.output, args.timeout,
            args.pos_tol, args.yaw_tol, args.grace)
        task.run()
    except rospy.ROSInterruptException:
        pass