#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
from std_msgs.msg import String  # Import String message type
from tf.transformations import euler_from_quaternion
import math
from datetime import datetime, timedelta
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
        self.path_timeout = 2.0  # Timeout in seconds for path topic updates
        self.realign_angle_threshold = math.radians(30)  # Angle threshold to trigger realignment

        # Subscribers
        rospy.Subscriber("/active_algorithm", String, self.active_algorithm_callback)
        rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.path_callback)
        rospy.Subscriber("/move_base/recovery_status", String, self.recovery_status_callback)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # State variables
        self.current_pose = None
        self.current_yaw = None
        self.path = None
        self.path_received = False
        self.current_waypoint_idx = 0  # Start at the first valid waypoint
        self.initial_alignment_done = False  # Alignment phase flag
        self.alignment_waypoints = None  # Fixed waypoints for initial alignment
        self.last_path_update = datetime.now()  # Track last time path was updated
        self.in_recovery_mode = False  # Track recovery mode status
        self.active_algorithm = "floam"  # Default algorithm
        self.odom_subscriber = None  # Placeholder for the odometry subscriber
        self.running = True  # Flag to control loop termination

        # Control loop rate
        self.rate = rospy.Rate(10)  # 10 Hz
        self.update_odom_subscription()
        # Attach signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle termination signals to shut down cleanly."""
        rospy.loginfo(f"Signal {signum} received. Stopping PathFollower.")
        self.running = False
        rospy.signal_shutdown("Shutting down PathFollower.")

    def recovery_status_callback(self, msg):
        """Update recovery mode status."""
        if "recovery" in msg.data.lower():
            rospy.loginfo("Move_base is in recovery mode.")
            self.in_recovery_mode = True
        else:
            self.in_recovery_mode = False

    def odom_callback(self, msg):
        """Update the robot's current pose and yaw."""
        self.current_pose = msg.pose.pose
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)


    def validate_odometry(self):
        """Validate if current odometry data is ready and consistent."""
        if self.current_pose is None or self.current_yaw is None:
            rospy.logwarn("Odometry data is incomplete. Waiting for updates.")
            return False
        return True


    def update_odom_subscription(self):
        """Update the odometry subscription based on the active algorithm."""
        if self.odom_subscriber:
            self.odom_subscriber.unregister()

        if self.active_algorithm == "floam":
            odom_topic = "/floam/odom"
        elif self.active_algorithm == "point_lio":
            odom_topic = "/point_lio/odom"
        else:
            rospy.logwarn(f"Unknown algorithm '{self.active_algorithm}'. Defaulting to floam.")
            odom_topic = "/floam/odom"

        self.odom_subscriber = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)
        rospy.loginfo(f"Subscribed to odometry topic: {odom_topic}")

        # Pause to allow odometry data to stabilize
        rospy.loginfo("Pausing to allow odometry data to stabilize.")
        rospy.sleep(0.5)  # Adjust the duration if needed



    def active_algorithm_callback(self, msg):
        """Callback for the active algorithm topic."""
        if msg.data != self.active_algorithm:
            rospy.loginfo(f"Switching active algorithm to: {msg.data}")
            self.active_algorithm = msg.data
            self.update_odom_subscription()

            # Reset state variables and wait for fresh odometry
            self.current_pose = None
            self.current_yaw = None
            rospy.loginfo("Waiting for fresh odometry after algorithm switch.")
            timeout_start = rospy.Time.now()
            while not self.validate_odometry():
                rospy.logwarn_throttle(2, "Waiting for valid odometry data...")
                rospy.sleep(0.1)
                if rospy.Time.now() - timeout_start > rospy.Duration(5.0):  # 5-second timeout
                    rospy.logerr("Timeout waiting for odometry data. Check the odometry source.")
                    break  # Exit the loop after timeout




    def path_callback(self, msg):
        """Receive and store the path."""
        self.path = msg.poses
        if len(self.path) > 0:
            self.path_received = True
            self.last_path_update = datetime.now()
            rospy.loginfo(f"Path updated with {len(self.path)} waypoints.")
            if self.initial_alignment_done and len(self.path) > 2:
                self.current_waypoint_idx = 2  # Skip the first two waypoints
        else:
            self.path_received = False  # Mark path as not received if empty
            rospy.logwarn("Received an empty path. Robot will stop if no path update follows.")

    def check_path_timeout(self):
        """Check if the path topic has timed out or if the path is empty."""
        time_since_update = datetime.now() - self.last_path_update

        # Check if the path topic has not been updated
        if time_since_update > timedelta(seconds=self.path_timeout):
            rospy.logwarn("No valid path for too long. Attempting recovery.")
            self.path_received = False

            # Check if move_base is in recovery mode
            if self.in_recovery_mode:
                rospy.loginfo("Move_base is in recovery mode. Rotating to refresh costmap...")
                self.perform_recovery_rotation()

            return True

        # Check if the path topic is explicitly empty
        if self.path is not None and len(self.path) == 0:
            rospy.logwarn("Path is empty. Attempting recovery.")
            self.path_received = False
            self.perform_recovery_rotation()
            return True

        return False

    def perform_recovery_rotation(self):
        """Perform left-right rotations to refresh costmap and facilitate path replanning."""
        rotation_duration = 0.5  # Duration of each rotation (seconds)
        twist = Twist()
        
        # Rotate left
        rospy.loginfo("Rotating left.")
        twist.angular.z = 0.3
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(rotation_duration)

        # Stop
        rospy.loginfo("Pausing rotation.")
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.2)

        # Rotate right
        rospy.loginfo("Rotating right.")
        twist.angular.z = -0.3
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(rotation_duration)

        # Stop
        rospy.loginfo("Pausing rotation.")
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.2)



    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


    def control_loop(self):
        """Main control loop for following the path."""
        while self.running and not rospy.is_shutdown():
            # Validate odometry data
            if not self.validate_odometry():
                rospy.logwarn_throttle(5, "Waiting for valid odometry data...")
                self.rate.sleep()
                continue
                
            if self.check_path_timeout():
                rospy.loginfo("Waiting for a valid path...")
                self.rate.sleep()
                continue

            if not self.path_received or not self.path:
                rospy.logwarn_throttle(5, "Path is empty. Stopping the robot.")
                self.publish_velocity(0.0, 0.0)
                self.rate.sleep()
                continue

            # Initial alignment phase or re-alignment phase if necessary
            if not self.initial_alignment_done:
                rospy.loginfo("Performing initial alignment with path.")
                self.initial_alignment_done = self.align_with_first_waypoint()
                self.rate.sleep()
                continue

            # Handle waypoint navigation as per existing logic
            self.navigate_to_waypoints()


            if self.current_waypoint_idx >= len(self.path):
                # Check if robot has reached the final goal
                if len(self.path) == 0:
                    rospy.logwarn("No valid goal. Path is empty.")
                    self.publish_velocity(0.0, 0.0)
                    continue

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

            # Trigger realignment if angle difference exceeds the threshold
            if abs(angle_diff) > self.realign_angle_threshold:
                rospy.logwarn("Angle difference exceeds threshold. Realigning...")
                realignment_done = self.align_with_first_waypoint()
                if not realignment_done:
                    rospy.logwarn("Re-alignment interrupted or failed. Retrying...")
                continue

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


    def align_with_first_waypoint(self):
        """Align the robot with the direction vector formed by the current path waypoints."""
        while not rospy.is_shutdown():
            # Dynamically update the alignment waypoints if the path changes
            if self.path and len(self.path) >= 4:
                waypoint3 = self.path[2].pose.position
                waypoint4 = self.path[3].pose.position
            else:
                rospy.logwarn("Path has less than 4 waypoints. Skipping alignment.")
                return True

            if self.current_pose is None or self.current_yaw is None:
                rospy.logwarn("Waiting for odometry data during alignment...")
                self.rate.sleep()
                continue

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
            twist.linear.x = 0.0
            self.cmd_vel_pub.publish(twist)

            # Check if a new path update suggests a drastic change, recheck waypoints
            if abs(angle_diff) > self.realign_angle_threshold:
                rospy.logwarn("Significant path change detected during alignment. Realigning to new path...")
                continue

            # Sleep to allow odometry updates
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
