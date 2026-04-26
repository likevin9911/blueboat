/**
 * goal_to_path_node.cpp  (v2 — feeds the search-bypass path)
 *
 * Test utility for driving the steering pipeline without a real search
 * planner. Subscribes to RViz "2D Nav Goal" on /move_base_simple/goal,
 * looks up the boat's current pose via tf, and publishes a 2-pose
 * nav_msgs/Path on ~waypoints (start, goal). The stitcher then runs a
 * single steering segment between them.
 *
 * If you want to feed A* / RRT* with the same RViz goal instead,
 * remap /move_base_simple/goal to /global_goal in the launch file.
 *
 * Subscribes:
 *   /move_base_simple/goal  (geometry_msgs/PoseStamped)
 *
 * Publishes:
 *   ~waypoints              (nav_msgs/Path) — 2 poses
 *
 * Params (private):
 *   ~global_frame  default "map"
 *   ~robot_frame   default "base_link"
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <memory>
#include <string>

class GoalToPath
{
public:
  GoalToPath()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    nhP.param<std::string>("global_frame", m_globalFrame, "map");
    nhP.param<std::string>("robot_frame",  m_robotFrame,  "base_link");

    m_pathPub = nhP.advertise<nav_msgs::Path>("waypoints", 10);
    m_goalSub = nh.subscribe("/move_base_simple/goal", 10,
                             &GoalToPath::goalCallback, this);

    m_tfListener = std::make_shared<tf2_ros::TransformListener>(m_tfBuffer);

    ROS_INFO("[goal_to_path] ready — click '2D Nav Goal' in RViz");
  }

private:
  void goalCallback(const geometry_msgs::PoseStamped& goal)
  {
    geometry_msgs::TransformStamped tf;
    try
    {
      tf = m_tfBuffer.lookupTransform(m_globalFrame, m_robotFrame,
                                       ros::Time(0.0), ros::Duration(1.0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("[goal_to_path] tf lookup failed: %s", ex.what());
      return;
    }

    double x0 = tf.transform.translation.x;
    double y0 = tf.transform.translation.y;
    double yaw0 = tf2::getYaw(tf.transform.rotation);
    double x1 = goal.pose.position.x;
    double y1 = goal.pose.position.y;

    if (std::hypot(x1 - x0, y1 - y0) < 0.1)
    {
      ROS_WARN("[goal_to_path] goal is on top of the robot, ignoring");
      return;
    }

    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = m_globalFrame;
    path.poses.resize(2);

    // Start pose: actual robot pose, with current heading
    path.poses[0].header = path.header;
    path.poses[0].pose.position.x = x0;
    path.poses[0].pose.position.y = y0;
    {
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw0);
      path.poses[0].pose.orientation.x = q.x();
      path.poses[0].pose.orientation.y = q.y();
      path.poses[0].pose.orientation.z = q.z();
      path.poses[0].pose.orientation.w = q.w();
    }

    // Goal pose: clicked position and clicked orientation
    path.poses[1].header = path.header;
    path.poses[1].pose.position.x = x1;
    path.poses[1].pose.position.y = y1;
    path.poses[1].pose.orientation = goal.pose.orientation;

    m_pathPub.publish(path);
    ROS_INFO("[goal_to_path] published 2-pose waypoint list (%.2f, %.2f) -> (%.2f, %.2f)",
             x0, y0, x1, y1);
  }

  ros::Publisher  m_pathPub;
  ros::Subscriber m_goalSub;
  tf2_ros::Buffer m_tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tfListener;
  std::string m_globalFrame;
  std::string m_robotFrame;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "goal_to_path_node");
  GoalToPath node;
  ros::spin();
  return 0;
}
