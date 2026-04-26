#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class Node:
    """
    Twist -> thruster command mixer for a boat.

    Subscribes:
      - cmd_vel (geometry_msgs/Twist)

    Publishes (Float32, clamped to [-1, 1]):
      - left_cmd
      - right_cmd
      - lateral_cmd (optional)

    Behavior:
      - Normal differential thrust: left = v - w, right = v + w
      - Point-turn override: if forward command is near zero but yaw is nonzero,
        force a point turn (left=-, right=+) so it spins in place.
    """

    def __init__(self,
                 linear_scaling: float,
                 angular_scaling: float,
                 lateral_scaling: float,
                 publish_lateral: bool,
                 v_dead: float,
                 w_dead: float,
                 min_point_turn: float):
        self.linear_scaling = float(linear_scaling)
        self.angular_scaling = float(angular_scaling)
        self.lateral_scaling = float(lateral_scaling)
        self.publish_lateral = bool(publish_lateral)

        # Deadbands are applied AFTER scaling (v_cmd, w_cmd)
        self.v_dead = float(v_dead)
        self.w_dead = float(w_dead)

        # Minimum thrust magnitude during point-turn override (0..1)
        self.min_point_turn = float(min_point_turn)

        self.left_pub = rospy.Publisher("left_cmd", Float32, queue_size=10)
        self.right_pub = rospy.Publisher("right_cmd", Float32, queue_size=10)
        self.left_msg = Float32()
        self.right_msg = Float32()

        if self.publish_lateral:
            self.lat_pub = rospy.Publisher("lateral_cmd", Float32, queue_size=10)
            self.lat_msg = Float32()
        else:
            self.lat_pub = None
            self.lat_msg = None

        rospy.Subscriber("cmd_vel", Twist, self.callback, queue_size=10)

    def callback(self, data: Twist):
        # Scale incoming commands
        v_cmd = self.linear_scaling * data.linear.x      # forward/back
        w_cmd = self.angular_scaling * data.angular.z    # yaw

        # Point-turn override:
        # If forward is ~0 but yaw is nonzero, force a pure spin.
        if abs(v_cmd) < self.v_dead and abs(w_cmd) > self.w_dead:
            s = 1.0 if w_cmd > 0 else -1.0
            turn_power = clamp(abs(w_cmd), 0.0, 1.0)
            if self.min_point_turn > 0.0:
                turn_power = max(self.min_point_turn, turn_power)

            left = -s * turn_power
            right = s * turn_power
        else:
            # Normal differential thrust
            left = v_cmd - w_cmd
            right = v_cmd + w_cmd

        # Clamp to [-1, 1]
        self.left_msg.data = clamp(left)
        self.right_msg.data = clamp(right)

        self.left_pub.publish(self.left_msg)
        self.right_pub.publish(self.right_msg)

        # Optional lateral (strafe): maps linear.y -> lateral_cmd
        if self.publish_lateral and self.lat_pub is not None and self.lat_msg is not None:
            vy_cmd = self.lateral_scaling * data.linear.y
            self.lat_msg.data = clamp(vy_cmd)
            self.lat_pub.publish(self.lat_msg)

        rospy.logdebug(
            "cmd_vel vx=%.3f vy=%.3f wz=%.3f -> v_cmd=%.3f w_cmd=%.3f -> L=%.3f R=%.3f%s",
            data.linear.x, data.linear.y, data.angular.z,
            v_cmd, w_cmd,
            self.left_msg.data, self.right_msg.data,
            (f" lat={self.lat_msg.data:.3f}" if self.publish_lateral and self.lat_msg else "")
        )


if __name__ == "__main__":
    rospy.init_node("twist2thrust", anonymous=True)

    # Params (tune these)
    linear_scaling = rospy.get_param("~linear_scaling", 0.6)
    angular_scaling = rospy.get_param("~angular_scaling", 3.0)

    # Lateral support (J/L -> linear.y)
    publish_lateral = rospy.get_param("~publish_lateral", True)
    lateral_scaling = rospy.get_param("~lateral_scaling", 0.6)

    # Point-turn behavior (deadbands are after scaling)
    v_dead = rospy.get_param("~v_dead", 0.05)
    w_dead = rospy.get_param("~w_dead", 0.05)
    min_point_turn = rospy.get_param("~min_point_turn", 0.5)

    rospy.loginfo(
        "twist2thrust params: linear_scaling=%.3f angular_scaling=%.3f publish_lateral=%s lateral_scaling=%.3f "
        "v_dead=%.3f w_dead=%.3f min_point_turn=%.3f",
        linear_scaling, angular_scaling, str(publish_lateral), lateral_scaling,
        v_dead, w_dead, min_point_turn
    )

    node = Node(
        linear_scaling=linear_scaling,
        angular_scaling=angular_scaling,
        lateral_scaling=lateral_scaling,
        publish_lateral=publish_lateral,
        v_dead=v_dead,
        w_dead=w_dead,
        min_point_turn=min_point_turn
    )

    rospy.spin()
