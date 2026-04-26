#!/usr/bin/env python
import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

class Figure8Manager:
    def __init__(self):  # ← FIXED: Double underscores
        rospy.init_node('figure8_manager')
        
        # Figure 8 Path
        # self.raw_points = [
        #     (18.5487, 1.0684, 0.0),
        #     (28.0015, 19.3756, 0.0),
        #     (49.4061, 20.7053, 0.0),
        #     (66.1470, 7.1973, 0.0),
        #     (93.4365, -16.6034, 0.0),
        #     (123.0372, -24.5691, 0.0),
        #     (139.2330, -24.0074, 0.0),
        #     (143.3049, -11.3510, 0.0),
        #     (133.9569, 13.8537, 0.0),
        #     (107.8714, 34.1238, 0.0),
        #     (87.9539, 25.9499, 0.0),
        #     (65.9586, -5.0476, 0.0),
        #     (39.7142, -4.2841, 0.0),
        #     (18.3151, 0.9366, 0.0)
        # ]

        # Square Path
        # self.raw_points = [
        #     (57.240234375, 0.18896484375, 1.52105712890625),
        #     (60.94545364379883, 4.375032424926758, 0.025482177734375),
        #     (100.97048950195312, -20.474014282226562, 1.4273223876953125),
        #     (106.52316284179688, -19.66790199279785, 0.02036285400390625),
        #     (110.75666809082031, -16.014802932739258, 0.01171112060546875),
        #     (112.11511993408203, -10.332133293151855, -0.02691650390625),
        #     (117.2966079711914, 22.506534576416016, 0.0455780029296875),
        #     (116.32504272460938, 27.71529197692871, -0.004058837890625),
        #     (113.5589599609375, 32.31767654418945, -0.01251220703125),
        #     (108.91693115234375, 34.04728698730469, 0.0098724365234375),
        #     (80.42095184326172, 46.272254943847656, 0.0159912109375),
        #     (75.16860961914062, 46.40375900268555, 0.03841400146484375),
        #     (70.8016357421875, 43.218624114990234, 0.025482177734375),
        #     (67.65713500976562, 37.3125114440918, 0.00675201416015625),
        #     (60.94545364379883, 4.375032424926758, 0.025482177734375),
        #     (61.56877899169922, -0.2776784896850586, 0.0171966552734375),
        #     (63.95226287841797, -5.2648091316223145, 0.00945281982421875),
        #     (69.39672088623047, -7.857223033905029, -0.00833892822265625),
        # ]

        # Cross Line Path
        self.raw_points = [
            (41.7664794921875, -3.8932647705078125, 0.001251220703125),
            (66.36786651611328, 1.6357345581054688, 0.09271240234375),
            (104.27783203125, 10.650741577148438, 0.0041046142578125),
            (128.72891235351562, 15.05129623413086, -0.033416748046875),
            (135.4423065185547, 10.695144653320312, 0.11383056640625),
            (138.42242431640625, 4.100826263427734, -0.0007171630859375),
            (138.20834350585938, -4.049842834472656, 0.0545654296875),
            (134.56781005859375, -16.325958251953125, 0.0821533203125),
            (114.95687866210938, -37.05326461791992, -0.0052642822265625),
            (105.28196716308594, -43.718841552734375, -0.08966064453125),
            (94.90403747558594, -36.98485565185547, 0.0262603759765625),
            (83.77713012695312, -10.397029876708984, -0.0487823486328125),
            (76.17150115966797, 14.219776153564453, -0.0462188720703125),
            (67.16551971435547, 41.02394485473633, 0.039154052734375),
        ]

        self.GOAL_DISTANCE_THRESHOLD = 3.5
        self.GOAL_ANGLE_THRESHOLD = 1.5
        
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.client.wait_for_server()
        rospy.loginfo("Connected!")
        
        self.current_pose = None
        self.odom_sub = rospy.Subscriber('/blueboat/robot_localization/odometry/filtered', Odometry, self.odom_callback, queue_size=1)
        rospy.sleep(2.0)
        
        self.mission_pub = rospy.Publisher('/mission_complete', Bool, queue_size=1)
        rospy.loginfo("Mission complete publisher ready on /mission_complete")
        
        rospy.loginfo(f"Loaded {len(self.raw_points)} waypoints")
        rospy.loginfo(f"Distance Threshold: {self.GOAL_DISTANCE_THRESHOLD}m")
        rospy.loginfo(f"Angle Threshold: {self.GOAL_ANGLE_THRESHOLD}rad")

    def odom_callback(self, msg):  
        self.current_pose = msg.pose.pose

    def get_quaternion(self, yaw): 
        return Quaternion(0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))

    def get_yaw_from_quaternion(self, q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def normalize_angle(self, angle):  
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_distance(self, x1, y1, x2, y2): 
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def send_goal(self, x, y, yaw):  
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(x, y, 0.0)
        goal.target_pose.pose.orientation = self.get_quaternion(yaw)
        
        self.client.send_goal(goal)
        rospy.loginfo(f"  → Goal sent: ({x:.2f}, {y:.2f})")

    def run(self):  
        rospy.loginfo("="*60)
        rospy.loginfo("Starting Figure 8 with Odometry-Based Goal Verification")
        rospy.loginfo("="*60)
        
        for i in range(len(self.raw_points)):
            x, y, _ = self.raw_points[i]
            
            next_i = (i + 1) % len(self.raw_points)
            nx, ny, _ = self.raw_points[next_i]
            target_yaw = math.atan2(ny - y, nx - x) #next waypoint
            
            if i == len(self.raw_points) - 1:
                target_yaw = math.atan2(y - self.raw_points[i-1][1], x - self.raw_points[i-1][0])

            rospy.loginfo(f"\n{'='*60}")
            rospy.loginfo(f"Waypoint {i+1}/{len(self.raw_points)}: ({x:.2f}, {y:.2f})")
            rospy.loginfo(f"{'='*60}")
            
            self.send_goal(x, y, target_yaw)
            
            rate = rospy.Rate(10)
            reached = False
            
            while not rospy.is_shutdown():
                if not self.current_pose:
                    rate.sleep()
                    continue

                dist_error = self.get_distance(
                    self.current_pose.position.x,
                    self.current_pose.position.y,
                    x, y
                )
                
                current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
                angle_error = self.normalize_angle(target_yaw - current_yaw)
                abs_angle_error = abs(angle_error)
                
                rospy.loginfo_throttle(1, f"  → Dist: {dist_error:.2f}m | Angle Err: {abs_angle_error:.2f}rad")
                
                if dist_error < self.GOAL_DISTANCE_THRESHOLD and abs_angle_error < self.GOAL_ANGLE_THRESHOLD:
                    rospy.loginfo(f"  → Goal Reached (Odometry Check)!")
                    reached = True
                    self.client.cancel_goal()
                    break
                
                state = self.client.get_state()
                if state == actionlib.GoalStatus.ABORTED:
                    rospy.logwarn("  → move_base aborted, but checking odometry for proximity...")
                
                rate.sleep()
            
            rospy.sleep(0.5)
        
        rospy.loginfo("\n" + "="*60)
        rospy.loginfo("All Waypoints Visited (Based on Odometry)")
        rospy.loginfo("="*60)
        
        rospy.loginfo("\n✅ Figure 8 COMPLETED SUCCESSFULLY (Odometry Verified)!")
        self.mission_pub.publish(Bool(True))
        
        rospy.loginfo("🚩 Mission complete signal published to /mission_complete")
        rospy.sleep(2.0)

if __name__ == '__main__': 
    try:
        Figure8Manager().run()
    except rospy.ROSInterruptException:
        pass