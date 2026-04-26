#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import math
from std_msgs.msg import String
import signal


class PathFollower:
    def __init__(self):
        rospy.init_node("path_follower")

        # Parameters
        self.linear_speed = 0.15  # Base linear speed
        self.max_angular_speed = 0.3  # Maximum angular speed
        self.angle_threshold = math.radians(5)  # Angular correction threshold (5 degrees)
        self.distance_tolerance = 0.1  # 10 cm tolerance for waypoint
        self.lookahead_distance = 0.5  # Distance to look ahead for smoother paths
        self.running = True  # Flag to control loop
        self.floam_odom = None
        self.point_lio_odom = None
        self.floam_odom_time = rospy.Time(0)
        self.point_lio_odom_time = rospy.Time(0)

        # Subscribers
        rospy.Subscriber("/floam/odom", Odometry, self.floam_odom_callback)
        rospy.Subscriber("/point_lio/odom", Odometry, self.point_lio_odom_callback)

        rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.path_callback)
        rospy.Subscriber("/active_algorithm", String, self.active_algorithm_callback)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # State variables
        self.current_pose = None
        self.current_yaw = None
        self.path = None
        self.path_received = False
        self.current_waypoint_idx = 0  # Start at the first valid waypoint
        self.initial_alignment_done = False  # Alignment phase flag
        self.active_algorithm = "floam"  # Default algorithm

        # Control loop rate
        self.rate = rospy.Rate(10)  # 10 Hz

        # Attach signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)


    def signal_handler(self, signum, frame):
        """Handle termination signals to shut down cleanly."""
        rospy.loginfo(f"Signal {signum} received. Stopping PathFollower.")
        self.running = False
        rospy.signal_shutdown("Shutting down PathFollower.")

    def floam_odom_callback(self, msg):
        """Stores the latest FLOAM odometry message."""
        self.floam_odom = msg
        self.floam_odom_time = rospy.Time.now()

    def point_lio_odom_callback(self, msg):
        """Stores the latest Point-LIO odometry message."""
        self.point_lio_odom = msg
        self.point_lio_odom_time = rospy.Time.now()



    def active_algorithm_callback(self, msg):
        """
        Just store which algorithm is active. No blocking, no forced wait.
        """
        if msg.data != self.active_algorithm:
            rospy.loginfo(f"Switching active algorithm to: {msg.data}")
            self.active_algorithm = msg.data

    def get_active_odom(self):
        """
        Return whichever Odometry message matches self.active_algorithm, if it is recent.
        Otherwise return None.
        """
        now = rospy.Time.now()
        freshness_threshold = rospy.Duration(1.0)  # 1 second, adjust if needed

        if self.active_algorithm == "floam":
            # Check if we have a FLOAM odom message & it's not too old
            if self.floam_odom and (now - self.floam_odom_time) < freshness_threshold:
                return self.floam_odom
            else:
                return None
        elif self.active_algorithm == "point_lio":
            # Check if we have a Point-LIO odom & it's not too old
            if self.point_lio_odom and (now - self.point_lio_odom_time) < freshness_threshold:
                return self.point_lio_odom
            else:
                return None
        else:
            rospy.logwarn_once(f"Unknown algorithm '{self.active_algorithm}'!")
            return None



    def path_callback(self, msg):
        """Receive and store the path."""
        if not self.path_received:
            self.path = msg.poses
            if len(self.path) <= 2:
                rospy.logwarn("Path has less than 3 waypoints. Cannot skip the first two.")
                self.current_waypoint_idx = 0
            else:
                self.current_waypoint_idx = 2  # Skip the first two waypoints
            self.path_received = True
            rospy.loginfo(f"Path received with {len(self.path)} waypoints.")


    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def align_with_first_waypoint(self):
        if len(self.path) < 4:
            rospy.logwarn("Path < 4 waypoints; skipping alignment.")
            return True

        # 3rd & 4th waypoints
        waypoint3 = self.path[2].pose.position
        waypoint4 = self.path[3].pose.position

        # Track last_yaw to detect no change
        last_yaw = None
        same_count = 0
        max_same_count = 20  # e.g. 3 seconds at 10 Hz

        while not rospy.is_shutdown():
            odom_msg = self.get_active_odom()
            if not odom_msg:
                rospy.logwarn_throttle(5, "No odom during alignment. Stopping.")
                self.publish_velocity(0, 0)
                self.rate.sleep()
                continue

            # Recompute current pose/yaw
            self.current_pose = odom_msg.pose.pose
            orientation_q = self.current_pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y,
                                orientation_q.z, orientation_q.w]
            _, _, self.current_yaw = euler_from_quaternion(orientation_list)

            if last_yaw is not None:
                if abs(self.current_yaw - last_yaw) < 1e-3:  # basically no change
                    same_count += 1
                else:
                    same_count = 0
            last_yaw = self.current_yaw

            if same_count > max_same_count:
                rospy.logwarn("Yaw hasn't changed for too long. Aborting alignment.")
                self.publish_velocity(0, 0)
                return True

            # Create the vector from the 3rd to the 4th waypoint
            vector_waypoint = [waypoint4.x - waypoint3.x, waypoint4.y - waypoint3.y]

            # Create the vector representing the robot's current heading
            robot_heading = [math.cos(self.current_yaw), math.sin(self.current_yaw)]

            # Compute the angle between the vectors using the dot product
            dot_product = (vector_waypoint[0] * robot_heading[0] +
                           vector_waypoint[1] * robot_heading[1])
            magnitude_waypoint = math.sqrt(vector_waypoint[0]**2 + vector_waypoint[1]**2)
            magnitude_robot = math.sqrt(robot_heading[0]**2 + robot_heading[1]**2)
            cos_angle = dot_product / (magnitude_waypoint * magnitude_robot)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to avoid numerical issues
            angle_diff = math.acos(cos_angle)

            # Determine the direction of rotation using the cross product
            cross_product = (vector_waypoint[0] * robot_heading[1] -
                             vector_waypoint[1] * robot_heading[0])
            if cross_product < 0:
                angle_diff = -angle_diff  # Negative angle indicates clockwise rotation

            rospy.loginfo(f"Current Yaw (from odom): {math.degrees(self.current_yaw):.2f} degrees")
            rospy.loginfo(f"Aligning with vector (3rd to 4th waypoint): Angle Difference: {math.degrees(angle_diff):.2f} degrees")

            # If the angle difference is within the threshold, stop and exit alignment
            if abs(angle_diff) <= self.angle_threshold:
                self.publish_velocity(0.0, 0.0)
                rospy.loginfo("Alignment with the vector complete.")
                return True  # Exit alignment and proceed to path following

            # Rotate in the correct direction
            twist = Twist()
            twist.angular.z = self.max_angular_speed * (1 if angle_diff > 0 else -1)
            self.cmd_vel_pub.publish(twist)

            # Sleep to allow odometry updates
            self.rate.sleep()


    def control_loop(self):
        """Main control loop for following the path."""
        while self.running and not rospy.is_shutdown():
            odom_msg = self.get_active_odom()
            if not odom_msg:
                # If the currently active odometry is invalid or stale, STOP
                rospy.logwarn_throttle(
                    5.0, 
                    f"No fresh odometry from '{self.active_algorithm}'. Robot stopping."
                )
                self.publish_velocity(0.0, 0.0)
                self.rate.sleep()
                continue

            # Extract current pose & yaw from the chosen odom
            self.current_pose = odom_msg.pose.pose
            orientation_q = self.current_pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y,
                                orientation_q.z, orientation_q.w]
            _, _, self.current_yaw = euler_from_quaternion(orientation_list)

            if not self.path_received or not self.path:
                rospy.logwarn_throttle(5, "Waiting for a path...")
                self.rate.sleep()
                continue

            # Initial alignment phase
            if not self.initial_alignment_done:
                self.initial_alignment_done = self.align_with_first_waypoint()
                self.rate.sleep()
                continue

            if self.current_waypoint_idx >= len(self.path):
                # Check if robot has reached the final goal
                final_goal = self.path[-1].pose.position
                dx = final_goal.x - self.current_pose.position.x
                dy = final_goal.y - self.current_pose.position.y
                distance_to_goal = math.sqrt(dx**2 + dy**2)

                if distance_to_goal <= self.distance_tolerance:
                    rospy.loginfo("Final goal reached. Stopping the robot.")
                    self.publish_velocity(0.0, 0.0)
                    self.path_received = False  # Wait for a new path
                    self.rate.sleep()
                    continue
                else:
                    rospy.loginfo(f"Approaching final goal. Distance: {distance_to_goal:.2f} m")
                    target_waypoint = final_goal
            else:
                # Find lookahead waypoint
                target_waypoint = None
                for i in range(self.current_waypoint_idx, len(self.path)):
                    waypoint = self.path[i].pose.position
                    dx = waypoint.x - self.current_pose.position.x
                    dy = waypoint.y - self.current_pose.position.y
                    distance = math.sqrt(dx**2 + dy**2)
                    if distance > self.lookahead_distance:
                        target_waypoint = waypoint
                        self.current_waypoint_idx = i
                        break

                if not target_waypoint:
                    rospy.logwarn("No suitable lookahead waypoint found. Correcting towards final goal.")
                    target_waypoint = self.path[-1].pose.position

            # Calculate distance and angle difference
            dx = target_waypoint.x - self.current_pose.position.x
            dy = target_waypoint.y - self.current_pose.position.y
            distance = math.sqrt(dx**2 + dy**2)
            desired_angle = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(desired_angle - self.current_yaw)

            rospy.loginfo(f"Current Yaw: {math.degrees(self.current_yaw):.2f} degrees")
            rospy.loginfo(f"Desired Angle to Waypoint: {math.degrees(desired_angle):.2f} degrees")
            rospy.loginfo(f"Angle Difference: {math.degrees(angle_diff):.2f} degrees")

            twist = Twist()

            # Adjust velocities
            if distance > self.distance_tolerance:
                if abs(angle_diff) > self.angle_threshold:
                    twist.angular.z = -max(min(self.max_angular_speed, abs(angle_diff)), -self.max_angular_speed) * (1 if angle_diff > 0 else -1)
                else:
                    twist.angular.z = 0.0

                twist.linear.x = self.linear_speed
                rospy.loginfo("Moving towards the waypoint with corrections if needed.")
            else:
                rospy.loginfo(f"Reached waypoint {self.current_waypoint_idx + 1}.")
                self.current_waypoint_idx += 1

            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()

    def publish_velocity(self, linear, angular):
        """Publish velocity commands."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)


if __name__ == "__main__":
    try:
        follower = PathFollower()
        follower.control_loop()
    except rospy.ROSInterruptException:
        pass
