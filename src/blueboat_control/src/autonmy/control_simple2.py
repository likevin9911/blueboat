#!/usr/bin/env python3
"""
Loitering Goal Navigator (hold position at goal with burst thruster pulses).

Behavior
- Navigate toward goal (optional; can be disabled).
- When within distance_tolerance of goal, enter LOITER:
  * Compute bearing to goal.
  * Choose the heading that's "parallel" to the goal line with MIN rotation
      - either facing the goal (bearing) OR facing away (bearing+pi).
  * If heading error > loiter_angle_tol -> apply short angular burst.
  * Else if distance error > loiter_hold_radius -> apply short linear burst
      in the boat's forward/back direction (sign chosen from current heading
      relative to the goal line).
  * Otherwise publish zero and wait. Repeat.

ROS Params (~)
  use_default_odom           (bool, default: True)
  default_odom_topic         (str,  default: "/odom")
  freshness_threshold_sec    (float, default: 1.0)

  # Basic nav to goal (before loiter)
  do_drive_to_goal           (bool,  default: True)
  linear_speed               (float, default: 0.15)      # m/s
  max_angular_speed          (float, default: 0.15)      # rad/s
  angle_threshold_deg        (float, default: 10.0)      # deg
  distance_tolerance         (float, default: 0.5)       # m

  # Loiter settings
  loiter_enable              (bool,  default: True)
  loiter_hold_radius         (float, default: 0.35)      # m (start pulsing if outside)
  loiter_angle_tol_deg       (float, default: 8.0)       # deg (align to parallel)
  loiter_linear_burst        (float, default: 0.25)      # m/s during pulse
  loiter_angular_burst       (float, default: 0.25)      # rad/s during pulse
  loiter_on_time             (float, default: 0.25)      # s burst time
  loiter_off_time            (float, default: 1.00)      # s rest time between bursts
  loiter_max_align_bursts    (int,   default: 10)        # safety cap per alignment attempt (0 = no cap)

Topics
  Sub: /move_base_simple/goal (geometry_msgs/PoseStamped)
  Sub: <odom>                 (nav_msgs/Odometry)
  Pub: /cmd_vel               (geometry_msgs/Twist)

Notes
- "Parallel with the goal": we align the boat's longitudinal axis along the line from
  boat→goal, choosing the direction (toward OR away) that requires the smallest rotation.
- If you want to align to the goal pose's orientation instead, flip the function
  _choose_parallel_heading() to use the goal yaw.

Author: you & ChatGPT
"""

import math
import signal
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

class GoalLoiter:
    def __init__(self):
        rospy.init_node("goal_loiter")

        # ---------------- Parameters ----------------
        self.use_default_odom = rospy.get_param("~use_default_odom", True)
        self.default_odom_topic = rospy.get_param("~default_odom_topic", "/odom")
        freshness_sec = float(rospy.get_param("~freshness_threshold_sec", 1.0))
        self.freshness_threshold = rospy.Duration(freshness_sec)

        self.do_drive_to_goal = rospy.get_param("~do_drive_to_goal", True)
        self.linear_speed = float(rospy.get_param("~linear_speed", 0.15))
        self.max_angular_speed = float(rospy.get_param("~max_angular_speed", 0.15))
        self.angle_threshold = math.radians(float(rospy.get_param("~angle_threshold_deg", 10.0)))
        self.distance_tolerance = float(rospy.get_param("~distance_tolerance", 0.5))

        self.loiter_enable = rospy.get_param("~loiter_enable", True)
        self.loiter_hold_radius = float(rospy.get_param("~loiter_hold_radius", 0.35))
        self.loiter_angle_tol = math.radians(float(rospy.get_param("~loiter_angle_tol_deg", 15.0)))
        self.loiter_linear_burst = float(rospy.get_param("~loiter_linear_burst", 0.15))
        self.loiter_angular_burst = float(rospy.get_param("~loiter_angular_burst", 0.15))
        self.loiter_on_time = float(rospy.get_param("~loiter_on_time", 1.5))
        self.loiter_off_time = float(rospy.get_param("~loiter_off_time", 1.00))
        self.loiter_max_align_bursts = int(rospy.get_param("~loiter_max_align_bursts", 0))

        self.rate = rospy.Rate(10)  # 10 Hz
        self.running = True

        # ---------------- State ----------------
        self.goal_xy = None           # (gx, gy)
        self.goal_yaw = None          # optional, if you want to use it later
        self.active_algorithm = "rtabmap"

        # Odom buffers + timestamps
        self.default_odom = None
        self.default_odom_time = rospy.Time(0)
        self.floam_odom = None
        self.floam_odom_time = rospy.Time(0)
        self.point_lio_odom = None
        self.point_lio_odom_time = rospy.Time(0)
        self.rtabmap_odom = None
        self.rtabmap_odom_time = rospy.Time(0)

        # ---------------- I/O ----------------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb)

        if self.use_default_odom:
            rospy.loginfo(f"[Loiter] Using single odom topic: {self.default_odom_topic}")
            rospy.Subscriber(self.default_odom_topic, Odometry, self.default_odom_cb)
        else:
            rospy.loginfo("[Loiter] Using odom switching across FLOAM / Point-LIO / RTAB-Map.")
            rospy.Subscriber("/floam/odom", Odometry, self.floam_cb)
            rospy.Subscriber("/point_lio/odom", Odometry, self.point_lio_cb)
            rospy.Subscriber("/rtabmap/odom", Odometry, self.rtabmap_cb)
            rospy.Subscriber("/active_algorithm", String, self.active_alg_cb)

        # Clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ---------------- Callbacks ----------------
    def _signal_handler(self, signum, frame):
        rospy.loginfo(f"[Loiter] Signal {signum} received. Stopping.")
        self.running = False
        self._zero()
        try:
            rospy.signal_shutdown("Shutting down GoalLoiter.")
        except Exception:
            pass

    def goal_cb(self, msg: PoseStamped):
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        # Cache goal yaw (in case you want to align to goal orientation later)
        q = msg.pose.orientation
        gx, gy, gz, gw = q.x, q.y, q.z, q.w
        _, _, yaw = euler_from_quaternion([gx, gy, gz, gw])
        self.goal_yaw = yaw
        rospy.loginfo(f"[Loiter] New goal -> x={self.goal_xy[0]:.2f}, y={self.goal_xy[1]:.2f}")

    def default_odom_cb(self, msg: Odometry):
        self.default_odom = msg
        self.default_odom_time = rospy.Time.now()

    def floam_cb(self, msg: Odometry):
        self.floam_odom = msg
        self.floam_odom_time = rospy.Time.now()

    def point_lio_cb(self, msg: Odometry):
        self.point_lio_odom = msg
        self.point_lio_odom_time = rospy.Time.now()

    def rtabmap_cb(self, msg: Odometry):
        self.rtabmap_odom = msg
        self.rtabmap_odom_time = rospy.Time.now()

    def active_alg_cb(self, msg: String):
        if msg.data != self.active_algorithm:
            rospy.loginfo(f"[Loiter] Switching active algorithm -> {msg.data}")
            self.active_algorithm = msg.data

    # ---------------- Helpers ----------------
    @staticmethod
    def _normalize(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _get_active_odom(self):
        now = rospy.Time.now()
        if self.use_default_odom:
            if self.default_odom and (now - self.default_odom_time) < self.freshness_threshold:
                return self.default_odom
            rospy.logwarn_throttle(5.0, f"[Loiter] No fresh odom on '{self.default_odom_topic}'.")
            return None

        # Switching mode
        if self.active_algorithm == "floam":
            if self.floam_odom and (now - self.floam_odom_time) < self.freshness_threshold:
                return self.floam_odom
        elif self.active_algorithm == "point_lio":
            if self.point_lio_odom and (now - self.point_lio_odom_time) < self.freshness_threshold:
                return self.point_lio_odom
        elif self.active_algorithm == "rtabmap":
            if self.rtabmap_odom and (now - self.rtabmap_odom_time) < self.freshness_threshold:
                return self.rtabmap_odom

        rospy.logwarn_throttle(5.0, f"[Loiter] No fresh odom for '{self.active_algorithm}'.")
        return None

    def _zero(self):
        self.cmd_pub.publish(Twist())

    def _pulse(self, twist: Twist, duration_s: float):
        """Publish a command for 'duration_s' then zero."""
        t_end = rospy.Time.now() + rospy.Duration(duration_s)
        rate = rospy.Rate(20)
        while self.running and not rospy.is_shutdown() and rospy.Time.now() < t_end:
            self.cmd_pub.publish(twist)
            rate.sleep()
        self._zero()

    # ---------------- Core logic ----------------
    def _choose_parallel_heading(self, bearing_to_goal, current_yaw):
        """
        Choose heading along the goal line (toward OR away) that minimizes rotation from current_yaw.
        Returns (target_heading, angle_error).
        """
        option_fwd = self._normalize(bearing_to_goal - current_yaw)
        option_rev = self._normalize(bearing_to_goal + math.pi - current_yaw)

        # Pick smaller magnitude rotation
        if abs(option_fwd) <= abs(option_rev):
            target = bearing_to_goal
            err = option_fwd
        else:
            target = self._normalize(bearing_to_goal + math.pi)
            err = option_rev
        return target, err

    def _rotate_to_goal_line(self):
        """Rotate to closest parallel heading to the goal line (short bursts)."""
        if not self.goal_xy:
            return

        bursts = 0
        while self.running and not rospy.is_shutdown():
            odom = self._get_active_odom()
            if not odom:
                self.rate.sleep()
                continue

            pose = odom.pose.pose
            q = pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            gx, gy = self.goal_xy
            cx, cy = pose.position.x, pose.position.y
            bearing = math.atan2(gy - cy, gx - cx)
            target, ang_err = self._choose_parallel_heading(bearing, yaw)

            rospy.loginfo_throttle(1.0, f"[Loiter/Align] target={math.degrees(target):.1f}°, "
                                        f"err={math.degrees(ang_err):.1f}°")

            if abs(ang_err) <= self.loiter_angle_tol:
                self._zero()
                return

            if self.loiter_max_align_bursts and (bursts >= self.loiter_max_align_bursts):
                rospy.logwarn("[Loiter/Align] Burst cap reached; proceeding anyway.")
                self._zero()
                return

            tw = Twist()
            tw.angular.z = self.loiter_angular_burst * ( -1.0 if ang_err > 0.0 else 1.0 )
            self._pulse(tw, self.loiter_on_time)
            rospy.sleep(self.loiter_off_time)
            bursts += 1

    def _loiter_hold(self):
        """Hold near the goal using bursty linear pulses once aligned."""
        if not self.goal_xy:
            return
        last_dist = None
        # We’ll loop indefinitely until a new goal comes or node stops.
        while self.running and not rospy.is_shutdown() and self.goal_xy is not None:
            odom = self._get_active_odom()
            if not odom:
                self.rate.sleep()
                continue

            pose = odom.pose.pose
            q = pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            gx, gy = self.goal_xy
            cx, cy = pose.position.x, pose.position.y
            dx, dy = gx - cx, gy - cy
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)

            # Keep parallel alignment fresh
            target, ang_err = self._choose_parallel_heading(bearing, yaw)
            if abs(ang_err) > self.loiter_angle_tol:
                # realign with an angular burst
                tw = Twist()
                tw.angular.z = self.loiter_angular_burst * ( -1.0 if ang_err > 0.0 else 1.0 )
                rospy.loginfo_throttle(1.0, f"[Loiter/Hold] Re-aligning (err={math.degrees(ang_err):.1f}°).")
                self._pulse(tw, self.loiter_on_time)
                rospy.sleep(self.loiter_off_time)
                continue

            # Within hold radius? (If not, nudge toward goal.)
            if dist > self.loiter_hold_radius:
                # Decide forward vs reverse burst based on heading vs bearing
                # If we're pointing within +/-90° toward the goal, push forward; else reverse.
                heading_to_goal = self._normalize(bearing - yaw)
                forward = abs(heading_to_goal) <= (math.pi / 2.0)

                tw = Twist()
                tw.linear.x = self.loiter_linear_burst * (1.0 if forward else -1.0)
                rospy.loginfo_throttle(1.0, f"[Loiter/Hold] Dist={dist:.2f} m -> "
                                            f"{'forward' if forward else 'reverse'} burst.")
                self._pulse(tw, self.loiter_on_time)
                rospy.sleep(self.loiter_off_time)
            else:
                # Inside the radius: stay still (small rest)
                rospy.loginfo_throttle(5.0, f"[Loiter/Hold] Holding (dist={dist:.2f} m).")
                self._zero()
                rospy.sleep(self.loiter_off_time)

            last_dist = dist

    def _drive_to_goal(self):
        """(Optional) Simple go-to-goal before loiter."""
        if not self.do_drive_to_goal or self.goal_xy is None:
            return

        while self.running and not rospy.is_shutdown() and self.goal_xy is not None:
            odom = self._get_active_odom()
            if not odom:
                self.rate.sleep()
                continue

            pose = odom.pose.pose
            q = pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            gx, gy = self.goal_xy
            cx, cy = pose.position.x, pose.position.y
            dx, dy = gx - cx, gy - cy
            dist = math.hypot(dx, dy)
            desired = math.atan2(dy, dx)
            ang_err = self._normalize(desired - yaw)

            rospy.loginfo_throttle(1.0, f"[Nav] Dist={dist:.2f} m | dYaw={math.degrees(ang_err):.1f}°")

            if dist <= self.distance_tolerance:
                rospy.loginfo("[Nav] Goal reached; entering LOITER.")
                self._zero()
                return

            tw = Twist()
            tw.linear.x = self.linear_speed
            if abs(ang_err) > self.angle_threshold:
                tw.angular.z = self.max_angular_speed * (-1.0 if ang_err > 0.0 else 1.0)

            self.cmd_pub.publish(tw)
            self.rate.sleep()

    # ---------------- Main loop ----------------
    def spin(self):
        rospy.loginfo("[Loiter] Ready. Click a goal in RViz to start.")
        while self.running and not rospy.is_shutdown():
            if self.goal_xy is None:
                rospy.logwarn_throttle(5.0, "[Loiter] Waiting for a goal...")
                self.rate.sleep()
                continue

            # Step 1: (optional) navigate close to the goal
            self._drive_to_goal()

            # Step 2: Loiter (align then hold with bursts)
            if self.loiter_enable and self.goal_xy is not None:
                rospy.loginfo("[Loiter] Aligning to goal line...")
                self._rotate_to_goal_line()
                rospy.loginfo("[Loiter] Holding station with burst pulses...")
                self._loiter_hold()

            # If we ever exit loiter (e.g., new goal), loop continues

        # safety
        self._zero()


if __name__ == "__main__":
    try:
        node = GoalLoiter()
        node.spin()
    except rospy.ROSInterruptException:
        pass
