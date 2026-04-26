#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion
import math
import signal


class GoalNavigator:
    def __init__(self):
        rospy.init_node("goal_navigator")

        # Parameters
        self.linear_speed = 0.15  # Base linear speed
        self.max_angular_speed = 0.15  # Maximum angular speed
        self.angle_threshold = math.radians(5)  # Angular correction threshold (5 degrees)
        self.distance_tolerance = 0.1  # 10 cm tolerance for goal
        self.running = True  # Flag to control loop
        self.floam_odom = None
        self.point_lio_odom = None
        self.rtabmap_odom = None
        self.floam_odom_time = rospy.Time(0)
        self.point_lio_odom_time = rospy.Time(0)
        self.rtabmap_odom_time = rospy.Time(0)

        # Subscribers
        rospy.Subscriber("/floam/odom", Odometry, self.floam_odom_callback)
        rospy.Subscriber("/point_lio/odom", Odometry, self.point_lio_odom_callback)
        rospy.Subscriber("/rtabmap/odom", Odometry, self.rtabmap_odom_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        rospy.Subscriber("/active_algorithm", String, self.active_algorithm_callback)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # State Variables
        self.current_pose = None
        self.current_yaw = None
        self.goal_position = None
        self.active_algorithm = "rtabmap"  # Default algorithm

        # Rate
        self.rate = rospy.Rate(10)  # 10 Hz

        # Attach signal handler for clean shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle termination signals to shut down cleanly."""
        rospy.loginfo(f"Signal {signum} received. Stopping GoalNavigator.")
        self.running = False
        rospy.signal_shutdown("Shutting down GoalNavigator.")

    def floam_odom_callback(self, msg):
        """Store the latest FLOAM odometry message."""
        self.floam_odom = msg
        self.floam_odom_time = rospy.Time.now()

    def point_lio_odom_callback(self, msg):
        """Store the latest Point-LIO odometry message."""
        self.point_lio_odom = msg
        self.point_lio_odom_time = rospy.Time.now()

    def rtabmap_odom_callback(self, msg):
        """Store the latest RTAB-Map odometry message."""
        self.rtabmap_odom = msg
        self.rtabmap_odom_time = rospy.Time.now()

    def active_algorithm_callback(self, msg):
        """Store the active algorithm."""
        if msg.data != self.active_algorithm:
            rospy.loginfo(f"Switching active algorithm to: {msg.data}")
            self.active_algorithm = msg.data

    def get_active_odom(self):
        """Return the active odometry message based on the selected algorithm."""
        now = rospy.Time.now()
        freshness_threshold = rospy.Duration(1.0)  # 1-second freshness threshold

        if self.active_algorithm == "floam":
            if self.floam_odom and (now - self.floam_odom_time) < freshness_threshold:
                return self.floam_odom
        elif self.active_algorithm == "point_lio":
            if self.point_lio_odom and (now - self.point_lio_odom_time) < freshness_threshold:
                return self.point_lio_odom
        elif self.active_algorithm == "rtabmap":
            if self.rtabmap_odom and (now - self.rtabmap_odom_time) < freshness_threshold:
                return self.rtabmap_odom

        rospy.logwarn_throttle(5, f"No fresh odometry data available for '{self.active_algorithm}'.")
        return None

    def goal_callback(self, msg):
        """Update the goal position."""
        self.goal_position = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"New goal received: x={self.goal_position[0]:.2f}, y={self.goal_position[1]:.2f}")

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def rotate_to_goal(self):
        """Rotate the robot to face the goal."""
        while not rospy.is_shutdown():
            odom_msg = self.get_active_odom()
            if not odom_msg:
                rospy.logwarn_throttle(5, "No valid odometry. Waiting...")
                self.rate.sleep()
                continue

            # Update robot pose and yaw
            self.current_pose = odom_msg.pose.pose
            orientation_q = self.current_pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            _, _, self.current_yaw = euler_from_quaternion(orientation_list)

            if self.goal_position is None:
                rospy.logwarn_throttle(5, "Waiting for a goal...")
                self.rate.sleep()
                continue

            # Calculate the angle to the goal
            goal_x, goal_y = self.goal_position
            current_x, current_y = self.current_pose.position.x, self.current_pose.position.y
            desired_angle = math.atan2(goal_y - current_y, goal_x - current_x)
            angle_diff = self.normalize_angle(desired_angle - self.current_yaw)

            rospy.loginfo(f"Desired Angle: {math.degrees(desired_angle):.2f}° | Angle Difference: {math.degrees(angle_diff):.2f}°")

            # Rotate to align with the goal
            twist = Twist()
            if abs(angle_diff) > self.angle_threshold:
                if angle_diff > 0:
                    twist.angular.z = -self.max_angular_speed  # Old behavior: rotate right
                else:
                    twist.angular.z = self.max_angular_speed  # Old behavior: rotate left
                self.cmd_vel_pub.publish(twist)
            else:
                rospy.loginfo("Aligned with the goal. Stopping rotation.")
                self.cmd_vel_pub.publish(Twist())  # Stop rotation
                break

            self.rate.sleep()



    def drive_to_goal(self):
        """Drive the robot toward the goal, adjusting angle while moving."""
        while not rospy.is_shutdown():
            odom_msg = self.get_active_odom()
            if not odom_msg:
                rospy.logwarn_throttle(5, "No valid odometry. Waiting...")
                self.rate.sleep()
                continue

            # Update robot pose and yaw
            self.current_pose = odom_msg.pose.pose
            orientation_q = self.current_pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            _, _, self.current_yaw = euler_from_quaternion(orientation_list)

            if self.goal_position is None:
                rospy.logwarn_throttle(5, "Waiting for a goal...")
                self.rate.sleep()
                continue

            # Calculate distance and angle to the goal
            goal_x, goal_y = self.goal_position
            current_x, current_y = self.current_pose.position.x, self.current_pose.position.y
            dx = goal_x - current_x
            dy = goal_y - current_y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            desired_angle = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(desired_angle - self.current_yaw)

            rospy.loginfo(f"Distance: {distance:.2f} m | Angle Difference: {math.degrees(angle_diff):.2f}°")

            # Stop if goal is reached
            if distance <= self.distance_tolerance:
                rospy.loginfo("Goal reached. Stopping movement.")
                self.goal_position = None
                self.cmd_vel_pub.publish(Twist())  # Stop the robot
                break

            # Prepare movement commands
            twist = Twist()
            twist.linear.x = self.linear_speed

            # Apply angular correction while moving
            if abs(angle_diff) > self.angle_threshold:
                twist.angular.z = self.max_angular_speed * (-1 if angle_diff > 0 else 1)

            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()




    def control_loop(self):
        while self.running and not rospy.is_shutdown():
            if self.goal_position is None:
                rospy.logwarn_throttle(5, "Waiting for a goal...")
                self.rate.sleep()
                continue

            rospy.loginfo("[INFO] Starting rotation to face the goal.")
            self.rotate_to_goal()

            rospy.loginfo("[INFO] Starting to drive toward the goal.")
            self.drive_to_goal()

    def publish_velocity(self, linear, angular):
        """Publish velocity commands."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)


if __name__ == "__main__":
    try:
        navigator = GoalNavigator()
        navigator.control_loop()
    except rospy.ROSInterruptException:
        pass
