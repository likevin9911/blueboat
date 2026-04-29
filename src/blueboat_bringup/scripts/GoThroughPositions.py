#!/usr/bin/env python3
"""
GoThroughPositions Task — acceptance-circle waypoint switching.

Rationale for switching scheme (literature):
============================================
Marine-vehicle path-following literature (Fossen & Pettersen 2014;
Lekkas & Fossen 2014; Liu et al. 2020 "Path Following Based on
Waypoints..."; Tandfonline 2025 "Maneuverability-based adaptive
LOS guidance") consistently uses the **acceptance circle** (also
called "circle of acceptance" or "switching radius") for waypoint
sequencing on underactuated surface vessels:

    "The straight path is switched to the next one when the AUV
     enters a circle of acceptance. The center of the circle is
     at p_{k+1}, and the radius is R_k."

This is a PREEMPTIVE switch — the next waypoint is activated as
soon as the vessel enters the radius around the *current* one,
**before** the controller has stopped. This avoids the
deceleration-acceleration cycle at every waypoint, which wastes
energy and produces large transient cross-track errors on
underactuated boats that can't stop precisely.

Acceptance radius selection:
    - Non-final waypoints: ~1-3× the boat's minimum turning radius.
      Larger radii give smoother trajectories but cut more corner.
      Lekkas & Fossen 2014 recommend radius >= turning_radius for
      feasibility; Liu 2020 uses 2-3 ship lengths.
    - Final waypoint: tight (e.g. 0.5 m) — this is where you actually
      want the boat to stop, and you accept the deceleration penalty.

The variable-radius extension (Tandfonline 2025) adapts the radius
per waypoint based on the angle between incoming and outgoing path
segments — sharp turns get smaller radii so the boat doesn't cut
the corner too aggressively. We expose --acceptance-radius and
--final-tolerance to control this; per-waypoint adaptive radii
are a future extension.

What the guidance node does on its end:
    With the matching guidance change, when the boat is "arrived"
    on the current path, guidance publishes (0, current_psi)
    continuously — it holds position. This task can preempt that
    at any time by publishing a new goal: guidance receives the
    new path, resets m_arrived = false, and resumes tracking.

Examples:
    rosrun blueboat_bringup GoThroughPositions.py \\
        --waypoints "10,0,0  20,5,90  25,5,0" \\
        --acceptance-radius 2.0 \\
        --final-tolerance 0.5 \\
        --output run.csv
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
# CONFIG
# ===========================================================================

DEFAULT_WAYPOINTS = "10,0,0  20,5,90  25.36,5,0"
DEFAULT_OUTPUT    = "multi.csv"
TIMEOUT           = 300.0
RECORD_HZ         = 10.0

# Literature-recommended starting points. Tune --acceptance-radius up for
# more open turns (smoother, more cut), down for tighter cornering.
ACCEPTANCE_RADIUS_M  = 2.0   # used for all NON-FINAL waypoints
FINAL_TOLERANCE_M    = 0.5   # used ONLY for the final waypoint
YAW_TOL_DEG          = 10.0  # only checked at final waypoint when --final-pose
GRACE_AFTER_ARRIVAL  = 10.0

GOAL_FRAME = "map"

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

TOPIC_GOAL                = "/move_base_simple/goal"
TOPIC_PLANNED_PATH        = "/planned_path"
TOPIC_PLANNED_CURVATURES  = "/planned_curvatures"
TOPIC_WAYPOINTS           = "/waypoints"

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
    "wp_index",
    "current_acceptance_radius",
    "arrived_final",
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
    if not s or not s.strip():
        raise ValueError("empty --waypoints string")
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
                 acceptance_radius, final_tolerance,
                 yaw_tol_deg, grace, arrival_source, final_pose_check):
        self.waypoints           = waypoints
        self.output              = output_path
        self.timeout             = timeout
        self.acceptance_radius   = acceptance_radius
        self.final_tolerance     = final_tolerance
        self.yaw_tol             = math.radians(yaw_tol_deg)
        self.grace               = grace
        self.arrival_source      = arrival_source
        self.final_pose_check    = final_pose_check  # if True, also gate on yaw at final wp
        self.lock                = threading.Lock()

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

        self._wp_index           = 0
        self._trajectory_done    = False
        self._completion_time    = None
        self._goal_pub           = None

        self._artifact_seen_waypoints = set()

    @staticmethod
    def _unpack_odom(msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        return [p.x, p.y, p.z, yaw,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y]

    def _radius_for_index(self, idx):
        """Return the acceptance radius for waypoint idx.
        Non-final waypoints use the wide acceptance circle (preemptive
        switch). Final waypoint uses the tight tolerance.
        """
        is_final = (idx == len(self.waypoints) - 1)
        return self.final_tolerance if is_final else self.acceptance_radius

    def _check_arrival(self, unpacked):
        """Acceptance-circle check. Caller must hold self.lock.
        Returns the next-goal index to publish (int) or None.
        """
        if self._trajectory_done:
            return None

        idx = self._wp_index
        if idx >= len(self.waypoints):
            self._trajectory_done = True
            self._completion_time = rospy.Time.now()
            return None

        wx, wy, wyaw = self.waypoints[idx]
        px, py, _, pyaw, _, _ = unpacked
        pos_err = math.hypot(px - wx, py - wy)

        radius     = self._radius_for_index(idx)
        is_final   = (idx == len(self.waypoints) - 1)

        if is_final and self.final_pose_check:
            yaw_err = abs(_wrap_pi(pyaw - wyaw))
            arrived = (pos_err < radius) and (yaw_err < self.yaw_tol)
            err_str = (f"pos_err={pos_err:.2f} m  "
                       f"yaw_err={math.degrees(yaw_err):.1f} deg  "
                       f"R={radius:.2f}")
        else:
            arrived = (pos_err < radius)
            err_str = f"pos_err={pos_err:.2f} m  R={radius:.2f}"

        if not arrived:
            return None

        new_idx = idx + 1
        self._wp_index = new_idx

        if new_idx >= len(self.waypoints):
            self._trajectory_done = True
            self._completion_time = rospy.Time.now()
            rospy.loginfo(
                f"[GoThrough] FINAL wp[{idx}] reached  {err_str}  "
                f"-> trajectory complete, recording {self.grace}s grace")
            return None

        rospy.loginfo(
            f"[GoThrough] wp[{idx}] entered acceptance circle  {err_str}  "
            f"-> preemptive switch to wp[{new_idx}]")
        return new_idx

    def cb_odom_filtered(self, msg):
        unpacked = self._unpack_odom(msg)
        publish_goal_for = None
        with self.lock:
            self._odom_filtered = unpacked
            if self.arrival_source == "ekf":
                publish_goal_for = self._check_arrival(unpacked)
        if publish_goal_for is not None:
            self._publish_goal(publish_goal_for)

    def cb_odom(self, msg):
        unpacked = self._unpack_odom(msg)
        publish_goal_for = None
        with self.lock:
            self._odom = unpacked
            if self.arrival_source == "carto":
                publish_goal_for = self._check_arrival(unpacked)
        if publish_goal_for is not None:
            self._publish_goal(publish_goal_for)

    def cb_p3d(self, msg):
        # Pure logging — Gazebo world frame, NOT used for arrival.
        unpacked = self._unpack_odom(msg)
        with self.lock:
            self._p3d = unpacked

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

    def _current_wp_index(self):
        with self.lock:
            return self._wp_index

    def cb_planned_path(self, msg):
        idx = self._current_wp_index()
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
        except Exception as e:
            rospy.logwarn(f"[GoThrough] failed to write planned_path: {e}")

    def cb_planned_curvatures(self, msg):
        idx = self._current_wp_index()
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
        except Exception as e:
            rospy.logwarn(f"[GoThrough] failed to write planned_curvatures: {e}")

    def cb_waypoints(self, msg):
        idx = self._current_wp_index()
        if idx in self._artifact_seen_waypoints:
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
            self._artifact_seen_waypoints.add(idx)
        except Exception as e:
            rospy.logwarn(f"[GoThrough] failed to write waypoints: {e}")

    def _write_row(self, event):
        with self.lock:
            current_radius = self._radius_for_index(
                min(self._wp_index, len(self.waypoints) - 1))
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
                + [self._wp_index, current_radius,
                   1 if self._trajectory_done else 0]
            )
        self._writer.writerow(row)

    def _publish_goal(self, idx):
        if self._goal_pub is None or idx >= len(self.waypoints):
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
        radius = self._radius_for_index(idx)
        rospy.loginfo(
            f"[GoThrough] goal[{idx}{'/final' if is_final else ''}] "
            f"-> x={x}  y={y}  yaw={math.degrees(yaw_rad):.1f} deg  "
            f"R={radius:.2f}")

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
        rospy.loginfo(
            f"[GoThrough] Recording -> {self.output}  @ {RECORD_HZ} Hz  "
            f"(acceptance-circle switching)")

        self._goal_pub = rospy.Publisher(TOPIC_GOAL, PoseStamped,
                                         queue_size=1, latch=True)
        rospy.sleep(1.0)

        n = len(self.waypoints)
        rospy.loginfo(f"[GoThrough] {n} waypoints (acceptance-circle scheme):")
        for i, (x, y, yaw_rad) in enumerate(self.waypoints):
            r = self._radius_for_index(i)
            tag = " (FINAL — tight tolerance)" if i == n - 1 else " (preemptive switch)"
            rospy.loginfo(
                f"  [{i}] x={x}  y={y}  yaw={math.degrees(yaw_rad):.1f} deg  "
                f"R={r:.2f} m{tag}")

        rospy.loginfo(
            f"[GoThrough] arrival rule: pos_err < acceptance_radius "
            f"(non-final R={self.acceptance_radius:.2f} m, "
            f"final R={self.final_tolerance:.2f} m"
            f"{', + yaw check' if self.final_pose_check else ''})  "
            f"source={self.arrival_source}")
        rospy.loginfo(
            f"[GoThrough] grace={self.grace}s, hard timeout={self.timeout}s")

        self._publish_goal(0)
        rospy.Timer(rospy.Duration(1.0 / RECORD_HZ), self._write_row)

        start = rospy.Time.now()
        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = (now - start).to_sec()

            if elapsed >= self.timeout:
                with self.lock:
                    done = self._trajectory_done
                    idx  = self._wp_index
                if done:
                    rospy.loginfo("[GoThrough] hard timeout reached during grace")
                else:
                    rospy.logwarn(
                        f"[GoThrough] hard timeout reached at wp[{idx}]/"
                        f"{len(self.waypoints)} (trajectory not complete)")
                break

            with self.lock:
                done       = self._trajectory_done
                completion = self._completion_time

            if done and completion is not None:
                if (now - completion).to_sec() >= self.grace:
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
        description="Drive through waypoints with acceptance-circle "
                    "switching (Fossen / Lekkas-style).",
    )
    parser.add_argument("--waypoints", type=str, default=DEFAULT_WAYPOINTS,
                        help="space- or semicolon-separated 'x,y,yaw_deg' tuples")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=TIMEOUT,
                        help="hard cap for whole sequence, seconds")
    parser.add_argument("--acceptance-radius", type=float,
                        default=ACCEPTANCE_RADIUS_M,
                        help="acceptance circle radius for NON-FINAL "
                             "waypoints, meters. Literature recommends "
                             "1-3x the boat's minimum turning radius. "
                             "Larger = smoother, more corner-cut. "
                             "Smaller = tighter cornering, sharper turns.")
    parser.add_argument("--final-tolerance", type=float,
                        default=FINAL_TOLERANCE_M,
                        help="position tolerance at the FINAL waypoint, "
                             "meters. This is where the boat actually "
                             "stops, so use the tightest value the boat "
                             "can reliably achieve given its momentum.")
    parser.add_argument("--final-pose", action="store_true",
                        help="also require yaw alignment at the final "
                             "waypoint (Lekkas/Fossen 'GoThroughPoses' "
                             "variant). Non-final waypoints are still "
                             "position-only.")
    parser.add_argument("--yaw-tol", type=float, default=YAW_TOL_DEG,
                        help="yaw tolerance at final waypoint, degrees "
                             "(only used with --final-pose)")
    parser.add_argument("--grace", type=float, default=GRACE_AFTER_ARRIVAL,
                        help="seconds to record after trajectory completes")
    parser.add_argument("--arrival-source", choices=["ekf", "carto"],
                        default="ekf",
                        help="which localizer drives the arrival check. "
                             "ekf=/odometry/filtered, carto=/odom. p3d "
                             "is logged but not used for arrival.")
    args = parser.parse_args()

    try:
        wps = parse_waypoints(args.waypoints)
    except ValueError as e:
        print(f"ERROR parsing --waypoints: {e}")
        raise SystemExit(2)

    try:
        task = GoThroughPositionsTask(
            wps, args.output, args.timeout,
            args.acceptance_radius, args.final_tolerance,
            args.yaw_tol, args.grace,
            args.arrival_source, args.final_pose)
        task.run()
    except rospy.ROSInterruptException:
        pass