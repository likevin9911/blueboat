#!/usr/bin/env python3
"""
GoToDock Task — autonomous docking via the new guidance/controller pipeline.

Architecture
============
This task replaces the C++ dock_approach_planner. It subscribes to the
existing dock perception nodes (adaptive_clustering -> l_shape_fitting,
which publish /dock_pose and /dock_dims) and converts those into two
waypoints — staging and dock — that drive through the same RPP guidance
+ controller stack as GoToPosition / GoToPose / GoThroughPositions.

Waypoint computation
--------------------
Given the locked dock pose D (position + long-axis orientation) and
dimensions (length L, width W), the task computes:

  side_unit  = +y_dock for port, -y_dock for starboard
  long_unit  = +x_dock                (long-axis direction)
  bow_unit   = +long_unit if bow_direction == "+x"
               -long_unit if bow_direction == "-x"
               -side_unit if bow_direction == "-y"

  dock_xy    = D.xy + (W/2 + inflation_radius + berth_standoff) * side_unit
                    + berth_long_offset * long_unit
  dock_yaw   = atan2(bow_unit.y, bow_unit.x)

  staging_xy = dock_xy - staging_distance * bow_unit
  staging_yaw = dock_yaw   (boat is already in final orientation; for
                           reverse-in, the path-search layer will plan
                           a reverse segment between staging and dock.)

The dock waypoint is the FINAL berth pose. Staging is intermediate.

Switching scheme (literature-grounded)
--------------------------------------
Acceptance-circle preemptive switching, per Liu et al. 2020 / Lekkas &
Fossen 2014 / Tandfonline 2025. The boat enters the staging acceptance
circle and the next goal (dock) is published immediately — no stop in
the middle, no deceleration-acceleration cycle.

  --staging-radius:  R for staging waypoint (preempt before stop)
  --dock-tolerance:  tight final tolerance at the dock berth

Dock pose locking
-----------------
The L-shape fitter publishes a pose every cluster cycle, with some
jitter. To avoid acting on a noisy estimate, the task waits for
--lock-frames consecutive detections (default 10), all within
--lock-radius meters of each other, then locks. The locked pose is
used to compute waypoints; further /dock_pose messages are logged
but do not change waypoints. (This mirrors lock_after_frames in the
original C++ planner.)

Metrics (literature-grounded; Mizuno et al. 2024, AUV docking surveys)
----------------------------------------------------------------------
  - success            : bool — pos_err < dock_tolerance AND
                                 yaw_err < yaw_tol AND
                                 min_clearance > 0 (no dock collision)
  - final_pos_err_m    : meters from boat to berth point at termination
  - final_yaw_err_deg  : degrees from commanded berth orientation
  - duration_s         : time from goal published to arrival latch
  - path_length_m      : integrated p3d distance
  - min_clearance_m    : closest approach to dock-bbox edge during run

Usage
-----
  rosrun blueboat_bringup GoToDock.py \\
      --dock-side starboard \\
      --bow-direction "-x" \\
      --staging-distance 8.0 \\
      --staging-radius 2.0 \\
      --berth-standoff 1.0 \\
      --inflation-radius 3.0 \\
      --output dock_run1.csv
"""

import rospy
import argparse
import csv
import math
import os
import threading
import time

from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32, Float32MultiArray
import tf.transformations as tft

from usv_msgs.msg import SpeedCourse


# ===========================================================================
# CONFIG
# ===========================================================================

DEFAULT_OUTPUT             = "dock_run.csv"
TIMEOUT                    = 300.0
RECORD_HZ                  = 10.0

# Dock geometry defaults — match your existing dock_approach_planner.
DEFAULT_DOCK_SIDE          = "starboard"          # "port" or "starboard"
DEFAULT_BOW_DIRECTION      = "-x"                 # "+x", "-x", "-y"
DEFAULT_STAGING_DISTANCE   = 8.0                  # meters
DEFAULT_BERTH_STANDOFF     = 1.0                  # meters beyond inflation
DEFAULT_INFLATION_RADIUS   = 3.0                  # must match map_inflating
DEFAULT_BERTH_LONG_OFFSET  = 0.0                  # along dock length

# Acceptance-circle / arrival
DEFAULT_STAGING_RADIUS     = 2.0                  # preemptive staging radius
DEFAULT_DOCK_TOLERANCE     = 0.5                  # tight final pos tol
DEFAULT_DOCK_YAW_TOL_DEG   = 15.0                 # final yaw tol (deg)

# Dock pose stability gating
DEFAULT_LOCK_FRAMES        = 10
DEFAULT_LOCK_RADIUS        = 0.5                  # meters

DEFAULT_DOCK_TIMEOUT       = 30.0   # max time to wait for dock detection
GRACE_AFTER_ARRIVAL        = 10.0

GOAL_FRAME = "map"

TOPIC_ODOM_FILTERED  = "/odometry/filtered"
TOPIC_ODOM           = "/odom"
TOPIC_P3D            = "/p3d_blueboat"
TOPIC_LEFT_THRUST    = "/left_thrust_cmd"
TOPIC_RIGHT_THRUST   = "/right_thrust_cmd"
TOPIC_SPEED_HEADING  = "/speed_heading"

TOPIC_DBG_SPEED_ACT  = "/debug/speed_actual"
TOPIC_DBG_SPEED_DES  = "/debug/speed_desired"
TOPIC_DBG_YAW_ACT    = "/debug/yaw_actual"
TOPIC_DBG_YAW_DES    = "/debug/yaw_desired"
TOPIC_DBG_THR_L      = "/debug/thrust_left"
TOPIC_DBG_THR_R      = "/debug/thrust_right"

TOPIC_GOAL                = "/move_base_simple/goal"
TOPIC_PLANNED_PATH        = "/planned_path"
TOPIC_PLANNED_CURVATURES  = "/planned_curvatures"
TOPIC_WAYPOINTS           = "/waypoints"

# Dock perception (from the existing C++ nodes)
TOPIC_DOCK_POSE           = "/dock_pose"
TOPIC_DOCK_DIMS           = "/dock_dims"

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
    "wp_index",         # 0 = staging, 1 = dock
    "current_acceptance_radius",
    "p3d_clearance_to_dock",
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


# ===========================================================================
# Dock geometry computation
# ===========================================================================

def _bow_unit(bow_direction, long_unit, side_unit):
    """Return the (ux, uy) unit vector for the boat's bow at the dock,
    given the dock's local +x (long) and +y (side-toward-water) axes.

    bow_direction:
        "+x" -> bow along +long_unit
        "-x" -> bow along -long_unit (boat reverses in)
        "-y" -> bow points away from wall (= +side_unit, water side);
                stern goes into dock
    """
    if bow_direction == "+x":
        return long_unit
    if bow_direction == "-x":
        return (-long_unit[0], -long_unit[1])
    if bow_direction == "-y":
        # bow points AWAY from dock wall, so along +side_unit (the unit
        # that points from dock centroid toward the water-side approach).
        return side_unit
    raise ValueError(f"unsupported bow_direction '{bow_direction}'")


def compute_dock_waypoints(dock_xy, dock_yaw, dock_length, dock_width,
                           dock_side, bow_direction,
                           berth_standoff, inflation_radius,
                           berth_long_offset, staging_distance):
    """Compute (staging_xy, staging_yaw, dock_xy, dock_yaw_final) in MAP frame.

    dock_xy/dock_yaw: locked dock centroid pose in map frame. dock_yaw
                     is the orientation of the dock's +x (long) axis.
    Returns:
        ((staging_x, staging_y), staging_yaw,
         (berth_x,   berth_y),   berth_yaw)
    """
    cx, cy = dock_xy
    cs, sn = math.cos(dock_yaw), math.sin(dock_yaw)
    # Dock-local axes expressed in map frame.
    long_unit = (cs, sn)             # +x_dock in map
    perp_unit = (-sn, cs)            # +y_dock in map (90° CCW from long)

    # Side unit: which of ±y_dock is the "water side" we approach from.
    # Convention from the C++ planner:
    #   port      -> +y_dock side (left wall of dock, facing water)
    #   starboard -> -y_dock side
    if dock_side == "port":
        side_unit = perp_unit
    elif dock_side == "starboard":
        side_unit = (-perp_unit[0], -perp_unit[1])
    else:
        raise ValueError(f"dock_side must be 'port' or 'starboard', got {dock_side}")

    # Berth point: out from the dock wall by half-width + inflation +
    # standoff, plus an optional shift along the dock length.
    standoff_total = (dock_width * 0.5) + inflation_radius + berth_standoff
    bx = cx + standoff_total * side_unit[0] + berth_long_offset * long_unit[0]
    by = cy + standoff_total * side_unit[1] + berth_long_offset * long_unit[1]

    # Bow direction at the berth, in MAP frame.
    bow_u = _bow_unit(bow_direction, long_unit, side_unit)
    berth_yaw = math.atan2(bow_u[1], bow_u[0])

    # Staging point: back along -bow_u by staging_distance. The boat's
    # bow at the berth points along bow_u; staging is "behind the bow",
    # so the boat moves along +bow_u from staging to berth (path-search
    # layer chooses forward or reverse based on direction sign).
    sx = bx - staging_distance * bow_u[0]
    sy = by - staging_distance * bow_u[1]

    # Staging orientation: same as berth orientation. The path planner
    # (Reeds-Shepp / HC-RS) will introduce a reverse segment if forward
    # alignment requires it.
    staging_yaw = berth_yaw

    return (sx, sy), staging_yaw, (bx, by), berth_yaw


# ===========================================================================

class GoToDockTask:

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()

        # Sensor caches
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

        # Dock pose locking
        self._dock_pose_history = []        # list of (x, y, yaw)
        self._dock_dims         = None      # (length, width)
        self._dock_locked       = False
        self._locked_dock       = None      # ((x, y), yaw)

        # Computed waypoints (in map frame)
        self._waypoints     = []            # [(x, y, yaw), (x, y, yaw)]
        self._wp_index      = 0
        self._goal_pub      = None

        # Run state
        self._t_start             = None
        self._goals_published_at  = None    # rospy.Time when waypoints set
        self._trajectory_done     = False
        self._completion_time     = None

        # Metrics
        self._prev_p3d_xy   = None
        self._path_length   = 0.0
        self._min_clearance = float("inf")  # closest approach to dock bbox

        # Artifact write tracking (per waypoint)
        self._artifact_seen_waypoints = set()

    # --------------------------------------------------------------------
    @staticmethod
    def _unpack_odom(msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        return [p.x, p.y, p.z, yaw,
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y]

    # --------------------------------------------------------------------
    # Sensor callbacks
    # --------------------------------------------------------------------
    # NOTE: cb_odom_filtered is defined further below — it does both
    # caching and staging-arrival preemption, so it lives near the
    # acceptance-circle logic.

    def cb_odom(self, msg):
        with self.lock: self._odom = self._unpack_odom(msg)

    def cb_p3d(self, msg):
        unpacked = self._unpack_odom(msg)
        with self.lock:
            self._p3d = unpacked
            px, py = unpacked[0], unpacked[1]

            # Path length integration
            if self._prev_p3d_xy is not None:
                dx = px - self._prev_p3d_xy[0]
                dy = py - self._prev_p3d_xy[1]
                self._path_length += math.hypot(dx, dy)
            self._prev_p3d_xy = (px, py)

            # Clearance to dock (only meaningful after lock)
            if self._dock_locked and self._dock_dims is not None:
                clearance = self._clearance_to_dock_bbox(px, py)
                if clearance < self._min_clearance:
                    self._min_clearance = clearance

            # Final waypoint arrival check (eval-only, drives task exit)
            if (self._dock_locked and self._wp_index >= len(self._waypoints) - 1
                    and not self._trajectory_done and self._waypoints):
                wx, wy, wyaw = self._waypoints[-1]
                _, _, _, pyaw, _, _ = unpacked
                pos_err = math.hypot(px - wx, py - wy)
                yaw_err = abs(_wrap_pi(pyaw - wyaw))
                if (pos_err < self.args.dock_tolerance
                        and yaw_err < math.radians(self.args.dock_yaw_tol)):
                    self._trajectory_done = True
                    self._completion_time = rospy.Time.now()
                    rospy.loginfo(
                        f"[GoToDock] DOCK ARRIVED  "
                        f"pos_err={pos_err:.2f} m  "
                        f"yaw_err={math.degrees(yaw_err):.1f} deg  "
                        f"min_clearance={self._min_clearance:.2f} m  "
                        f"-> recording {self.args.grace}s grace")

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

    # --------------------------------------------------------------------
    # Dock perception callbacks
    # --------------------------------------------------------------------
    def cb_dock_dims(self, msg):
        # Vector3: x = length (along long axis), y = width
        with self.lock:
            self._dock_dims = (msg.x, msg.y)

    def cb_dock_pose(self, msg):
        if msg.header.frame_id and msg.header.frame_id != GOAL_FRAME:
            rospy.logwarn_throttle(5.0,
                f"[GoToDock] /dock_pose frame is "
                f"'{msg.header.frame_id}', expected '{GOAL_FRAME}' — "
                f"check the dock estimator's target_frame param")
            return

        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = _yaw_from_quat(msg.pose.orientation)

        publish_now = False
        with self.lock:
            if self._dock_locked:
                # We've already locked. Log only.
                return

            self._dock_pose_history.append((x, y, yaw))
            # Keep only the last N
            if len(self._dock_pose_history) > self.args.lock_frames * 2:
                self._dock_pose_history.pop(0)

            # Check stability: last lock_frames samples within lock_radius
            if len(self._dock_pose_history) >= self.args.lock_frames:
                window = self._dock_pose_history[-self.args.lock_frames:]
                xs = [p[0] for p in window]
                ys = [p[1] for p in window]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                max_dev = max(math.hypot(p[0] - cx, p[1] - cy)
                              for p in window)
                if max_dev < self.args.lock_radius:
                    # Stable. Lock to the mean position, mean yaw.
                    # Mean of yaw needs angle averaging.
                    sin_sum = sum(math.sin(p[2]) for p in window)
                    cos_sum = sum(math.cos(p[2]) for p in window)
                    cyaw = math.atan2(sin_sum, cos_sum)
                    self._locked_dock = ((cx, cy), cyaw)
                    self._dock_locked = True
                    publish_now = True
                    rospy.loginfo(
                        f"[GoToDock] dock LOCKED after {self.args.lock_frames}"
                        f" stable frames: x={cx:.2f} y={cy:.2f} "
                        f"yaw={math.degrees(cyaw):.1f} deg "
                        f"(stability={max_dev:.2f} m)")

        if publish_now:
            # Compute waypoints and publish first goal — outside the lock
            # because publishing is a blocking ROS op.
            self._setup_waypoints_and_publish_first()

    # --------------------------------------------------------------------
    def _clearance_to_dock_bbox(self, px, py):
        """Distance from point (px, py) to the dock's oriented bounding
        box. 0 if inside; positive otherwise."""
        if not self._dock_locked or self._dock_dims is None:
            return float("inf")
        (cx, cy), yaw = self._locked_dock
        L, W = self._dock_dims
        # Transform the point into dock-local frame.
        cs, sn = math.cos(-yaw), math.sin(-yaw)
        rx = cs * (px - cx) - sn * (py - cy)
        ry = sn * (px - cx) + cs * (py - cy)
        # Distance to bbox extents (half-extents L/2 along x, W/2 along y).
        dx = max(0.0, abs(rx) - L * 0.5)
        dy = max(0.0, abs(ry) - W * 0.5)
        return math.hypot(dx, dy)

    # --------------------------------------------------------------------
    def _setup_waypoints_and_publish_first(self):
        if self._dock_dims is None:
            rospy.logwarn(
                "[GoToDock] dock locked but /dock_dims not received yet — "
                "using estimator output without size info; using "
                "min/max dock width from args is not implemented in this "
                "wrapper, expecting /dock_dims. Aborting.")
            return

        (dx, dy), dyaw = self._locked_dock
        L, W = self._dock_dims

        try:
            (sx, sy), syaw, (bx, by), byaw = compute_dock_waypoints(
                (dx, dy), dyaw, L, W,
                self.args.dock_side, self.args.bow_direction,
                self.args.berth_standoff, self.args.inflation_radius,
                self.args.berth_long_offset, self.args.staging_distance)
        except ValueError as e:
            rospy.logerr(f"[GoToDock] waypoint computation failed: {e}")
            return

        with self.lock:
            self._waypoints = [(sx, sy, syaw), (bx, by, byaw)]
            self._wp_index = 0
            self._goals_published_at = rospy.Time.now()

        rospy.loginfo(
            f"[GoToDock] waypoints (acceptance-circle scheme):")
        rospy.loginfo(
            f"  [0] STAGING  x={sx:.2f}  y={sy:.2f}  "
            f"yaw={math.degrees(syaw):.1f} deg  "
            f"R={self.args.staging_radius:.2f} m (preempt)")
        rospy.loginfo(
            f"  [1] DOCK     x={bx:.2f}  y={by:.2f}  "
            f"yaw={math.degrees(byaw):.1f} deg  "
            f"R={self.args.dock_tolerance:.2f} m (final)")

        self._publish_goal(0)

    # --------------------------------------------------------------------
    def _publish_goal(self, idx):
        if self._goal_pub is None:
            return
        with self.lock:
            if idx >= len(self._waypoints):
                return
            x, y, yaw = self._waypoints[idx]

        goal = PoseStamped()
        goal.header.frame_id = GOAL_FRAME
        goal.header.stamp    = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        q = tft.quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]
        self._goal_pub.publish(goal)
        is_final = (idx == len(self._waypoints) - 1)
        radius = (self.args.dock_tolerance if is_final
                  else self.args.staging_radius)
        rospy.loginfo(
            f"[GoToDock] goal[{idx}{'/dock' if is_final else '/staging'}] "
            f"-> x={x:.2f}  y={y:.2f}  yaw={math.degrees(yaw):.1f} deg  "
            f"R={radius:.2f}")

    # --------------------------------------------------------------------
    # EKF arrival check for STAGING preemption (acceptance-circle).
    # The DOCK arrival is checked in cb_p3d for evaluation.
    # --------------------------------------------------------------------
    def _check_staging_arrival(self, ekf_unpacked):
        with self.lock:
            if not self._dock_locked or not self._waypoints:
                return None
            if self._wp_index >= len(self._waypoints) - 1:
                return None  # already on final
            if self._trajectory_done:
                return None
            wx, wy, _ = self._waypoints[self._wp_index]
            px, py = ekf_unpacked[0], ekf_unpacked[1]
            pos_err = math.hypot(px - wx, py - wy)
            if pos_err < self.args.staging_radius:
                self._wp_index += 1
                rospy.loginfo(
                    f"[GoToDock] staging acceptance circle entered  "
                    f"pos_err={pos_err:.2f} m  R={self.args.staging_radius:.2f} m  "
                    f"-> preemptive switch to dock waypoint")
                return self._wp_index
        return None

    # Override EKF callback so it also drives staging preemption.
    def cb_odom_filtered(self, msg):
        unpacked = self._unpack_odom(msg)
        publish_for = None
        with self.lock:
            self._odom_filtered = unpacked
        publish_for = self._check_staging_arrival(unpacked)
        if publish_for is not None:
            self._publish_goal(publish_for)

    # --------------------------------------------------------------------
    # Per-goal artifacts
    # --------------------------------------------------------------------
    def _current_wp_index(self):
        with self.lock:
            return self._wp_index

    def cb_planned_path(self, msg):
        idx = self._current_wp_index()
        path_csv = f"{_stem(self.args.output)}_wp{idx}_planned_path.csv"
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
                f"[GoToDock] wp[{idx}] planned_path: "
                f"{len(msg.poses)} poses -> {path_csv}")
        except Exception as e:
            rospy.logwarn(f"[GoToDock] failed to write planned_path: {e}")

    def cb_planned_curvatures(self, msg):
        idx = self._current_wp_index()
        curv_csv = f"{_stem(self.args.output)}_wp{idx}_planned_curvatures.csv"
        try:
            with open(curv_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["i", "kappa"])
                for i, k in enumerate(msg.data):
                    w.writerow([i, k])
            rospy.loginfo(
                f"[GoToDock] wp[{idx}] planned_curvatures: "
                f"{len(msg.data)} samples -> {curv_csv}")
        except Exception as e:
            rospy.logwarn(f"[GoToDock] failed to write planned_curvatures: {e}")

    def cb_waypoints(self, msg):
        idx = self._current_wp_index()
        if idx in self._artifact_seen_waypoints:
            return
        wp_csv = f"{_stem(self.args.output)}_wp{idx}_waypoints.csv"
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
            self._artifact_seen_waypoints.add(idx)
        except Exception as e:
            rospy.logwarn(f"[GoToDock] failed to write waypoints: {e}")

    # --------------------------------------------------------------------
    def _radius_for_index(self, idx):
        if idx >= len(self._waypoints) - 1:
            return self.args.dock_tolerance
        return self.args.staging_radius

    def _write_row(self, event):
        with self.lock:
            current_radius = (self._radius_for_index(self._wp_index)
                              if self._waypoints else 0.0)
            clearance = (self._clearance_to_dock_bbox(
                            self._p3d[0], self._p3d[1])
                         if self._dock_locked and self._p3d[0] is not None
                         else 0.0)
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
                + [self._wp_index, current_radius, clearance,
                   1 if self._trajectory_done else 0]
            )
        self._writer.writerow(row)

    # --------------------------------------------------------------------
    def _write_summary(self):
        """Literature-standard docking metrics (Mizuno et al. 2024;
        AUV docking surveys)."""
        summary_csv = f"{_stem(self.args.output)}_summary.csv"
        with self.lock:
            success_pos_yaw = self._trajectory_done
            no_collision    = self._min_clearance > 0.0
            success         = success_pos_yaw and no_collision
            final_x         = self._p3d[0]
            final_y         = self._p3d[1]
            final_yaw       = self._p3d[3]
            duration        = ((self._completion_time - self._goals_published_at).to_sec()
                               if (success_pos_yaw and self._goals_published_at
                                   and self._completion_time) else None)
            path_len        = self._path_length
            min_clear       = (self._min_clearance
                               if self._min_clearance != float("inf") else None)
            berth_x = berth_y = berth_yaw = None
            if self._waypoints:
                berth_x, berth_y, berth_yaw = self._waypoints[-1]

        if final_x is None or berth_x is None:
            final_pos_err = None
            final_yaw_err = None
        else:
            final_pos_err = math.hypot(final_x - berth_x, final_y - berth_y)
            final_yaw_err = math.degrees(abs(_wrap_pi(final_yaw - berth_yaw)))

        try:
            with open(summary_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["success", "no_collision",
                            "final_pos_err_m", "final_yaw_err_deg",
                            "duration_s", "path_length_m",
                            "min_clearance_m"])
                w.writerow([
                    int(success), int(no_collision),
                    f"{final_pos_err:.3f}" if final_pos_err is not None else "",
                    f"{final_yaw_err:.3f}" if final_yaw_err is not None else "",
                    f"{duration:.3f}"      if duration      is not None else "",
                    f"{path_len:.3f}",
                    f"{min_clear:.3f}"     if min_clear     is not None else "",
                ])
            rospy.loginfo(
                f"[GoToDock] summary: success={success}  "
                f"no_collision={no_collision}  "
                f"pos_err={final_pos_err}  yaw_err={final_yaw_err}  "
                f"duration={duration}  path_len={path_len:.2f}  "
                f"min_clear={min_clear}  -> {summary_csv}")
        except Exception as e:
            rospy.logwarn(f"[GoToDock] failed to write summary: {e}")

    # --------------------------------------------------------------------
    def run(self):
        rospy.init_node("go_to_dock_task", anonymous=True)

        # Sensors
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

        # Dock perception (the C++ nodes still publish these)
        rospy.Subscriber(TOPIC_DOCK_POSE, PoseStamped, self.cb_dock_pose)
        rospy.Subscriber(TOPIC_DOCK_DIMS, Vector3,     self.cb_dock_dims)

        self._csvfile = open(self.args.output, "w", newline="")
        self._writer  = csv.writer(self._csvfile)
        self._writer.writerow(CSV_HEADER)

        self._goal_pub = rospy.Publisher(TOPIC_GOAL, PoseStamped,
                                         queue_size=1, latch=True)
        rospy.sleep(1.0)

        rospy.loginfo(
            f"[GoToDock] Recording -> {self.args.output}  @ {RECORD_HZ} Hz")
        rospy.loginfo(
            f"[GoToDock] dock_side={self.args.dock_side}  "
            f"bow_direction={self.args.bow_direction}  "
            f"staging_distance={self.args.staging_distance}")
        rospy.loginfo(
            f"[GoToDock] berth_standoff={self.args.berth_standoff}  "
            f"inflation_radius={self.args.inflation_radius}  "
            f"berth_long_offset={self.args.berth_long_offset}")
        rospy.loginfo(
            f"[GoToDock] acceptance: staging R={self.args.staging_radius:.2f} m, "
            f"dock R={self.args.dock_tolerance:.2f} m + yaw<"
            f"{self.args.dock_yaw_tol:.1f} deg")
        rospy.loginfo(
            f"[GoToDock] waiting up to {self.args.dock_timeout}s for stable "
            f"dock detection (lock_frames={self.args.lock_frames}, "
            f"lock_radius={self.args.lock_radius})")

        self._t_start = rospy.Time.now()
        rospy.Timer(rospy.Duration(1.0 / RECORD_HZ), self._write_row)

        rate = rospy.Rate(20.0)
        dock_timeout_at = self._t_start + rospy.Duration(self.args.dock_timeout)
        warned_no_dock = False

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = (now - self._t_start).to_sec()

            # If dock never locks, abort with a warning
            with self.lock:
                locked = self._dock_locked
            if not locked and now > dock_timeout_at and not warned_no_dock:
                rospy.logwarn(
                    f"[GoToDock] no stable dock pose after "
                    f"{self.args.dock_timeout}s — aborting")
                warned_no_dock = True
                break

            if elapsed >= self.args.timeout:
                with self.lock:
                    done = self._trajectory_done
                if done:
                    rospy.loginfo("[GoToDock] hard timeout reached during grace")
                else:
                    rospy.logwarn(
                        f"[GoToDock] hard timeout reached without arrival "
                        f"(t={elapsed:.1f}s, locked={locked})")
                break

            with self.lock:
                done       = self._trajectory_done
                completion = self._completion_time

            if done and completion is not None:
                if (now - completion).to_sec() >= self.args.grace:
                    rospy.loginfo(
                        f"[GoToDock] grace window complete (t={elapsed:.1f}s)")
                    break

            rate.sleep()

        self._csvfile.flush()
        self._csvfile.close()
        self._write_summary()
        rospy.loginfo(f"[GoToDock] Done — saved {self.args.output}")


# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Autonomous docking via the new RPP guidance stack. "
                    "Wraps the existing C++ dock perception (clustering + "
                    "L-shape fitting), adds Python task logic with "
                    "acceptance-circle waypoint switching.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=TIMEOUT,
                        help="hard cap for whole task, seconds")
    parser.add_argument("--dock-timeout", type=float, default=DEFAULT_DOCK_TIMEOUT,
                        help="seconds to wait for stable dock detection")
    parser.add_argument("--grace", type=float, default=GRACE_AFTER_ARRIVAL,
                        help="seconds to record after dock arrival")

    # Dock geometry
    parser.add_argument("--dock-side", choices=["port", "starboard"],
                        default=DEFAULT_DOCK_SIDE,
                        help="which side of the dock to approach. port = "
                             "+y_dock face (boat to dock's left-water-side); "
                             "starboard = -y_dock face.")
    parser.add_argument("--bow-direction", choices=["+x", "-x", "-y"],
                        default=DEFAULT_BOW_DIRECTION,
                        help="bow orientation at the berth, in dock frame. "
                             "+x = bow along dock long axis; "
                             "-x = bow against dock long axis (reverse-in); "
                             "-y = bow away from wall (stern-in).")
    parser.add_argument("--staging-distance", type=float,
                        default=DEFAULT_STAGING_DISTANCE,
                        help="meters between staging and dock waypoint")
    parser.add_argument("--berth-standoff", type=float,
                        default=DEFAULT_BERTH_STANDOFF,
                        help="meters beyond inflated map boundary at berth")
    parser.add_argument("--inflation-radius", type=float,
                        default=DEFAULT_INFLATION_RADIUS,
                        help="must match map_inflating's inflation_radius")
    parser.add_argument("--berth-long-offset", type=float,
                        default=DEFAULT_BERTH_LONG_OFFSET,
                        help="shift along dock length from centre, meters")

    # Acceptance-circle / arrival
    parser.add_argument("--staging-radius", type=float,
                        default=DEFAULT_STAGING_RADIUS,
                        help="acceptance circle radius for staging waypoint "
                             "(preemptive switch). Larger = smoother but "
                             "more corner-cut; smaller = tighter cornering.")
    parser.add_argument("--dock-tolerance", type=float,
                        default=DEFAULT_DOCK_TOLERANCE,
                        help="position tolerance at the dock berth")
    parser.add_argument("--dock-yaw-tol", type=float,
                        default=DEFAULT_DOCK_YAW_TOL_DEG,
                        help="yaw tolerance at the dock berth, degrees")

    # Dock pose locking
    parser.add_argument("--lock-frames", type=int,
                        default=DEFAULT_LOCK_FRAMES,
                        help="number of consecutive stable detections "
                             "required to lock the dock pose")
    parser.add_argument("--lock-radius", type=float,
                        default=DEFAULT_LOCK_RADIUS,
                        help="max position deviation for stability check, m")

    args = parser.parse_args()

    try:
        task = GoToDockTask(args)
        task.run()
    except rospy.ROSInterruptException:
        pass