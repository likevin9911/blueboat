/**
 * dubins_planner.cpp
 *
 * Steering function: Dubins (G1 continuous, forwards-only).
 *
 * I/O contract (shared by all four planner nodes in this package):
 *   Subscribes: ~planning_input  (blueboat_navigation/DubinInput)
 *   Publishes:  ~planned_path    (nav_msgs/Path)
 *
 * Driving direction encoding:
 *   The steering_functions library populates state.d ∈ {-1, 0, +1} for every
 *   pose along the path (forward, transition/cusp, reverse). To bypass any
 *   forward/reverse heuristic in the downstream guidance/controller, we
 *   embed state.d into pose.position.z of each PoseStamped. The 2D guidance
 *   node should read pose.position.z directly to decide commanded sign of
 *   surge speed. For the pure Dubins planner this is always +1.
 *
 * Params (private namespace):
 *   ~turning_radius   [m]   -> kappa_max = 1 / turning_radius        (default 1.0)
 *   ~path_resolution  [m]   -> discretization between integrated states (default 0.05)
 *   ~forwards_only    [bool]-> exposed for completeness; Dubins is forwards by definition (default true)
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#include <blueboat_navigation/DubinInput.h>

#include "steering_functions/dubins_state_space/dubins_state_space.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace blueboat_navigation
{

class DubinsPlanner
{
public:
  DubinsPlanner()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    double turning_radius = nhP.param("turning_radius", 1.0);
    m_kappaMax       = 1.0 / turning_radius;
    m_discretization = nhP.param("path_resolution", 0.05);
    m_forwards       = nhP.param("forwards_only", true);

    m_stateSpace = std::make_shared<steering::Dubins_State_Space>(
        m_kappaMax, m_discretization, m_forwards);

    m_pathPub  = nhP.advertise<nav_msgs::Path>("planned_path", 10);
    m_inputSub = nhP.subscribe("planning_input", 10,
                               &DubinsPlanner::onInput, this);

    ROS_INFO("[dubins_planner] kappa_max=%.3f (R=%.2f m)  res=%.3f m  forwards=%s",
             m_kappaMax, turning_radius, m_discretization,
             m_forwards ? "true" : "false");
  }

private:
  void onInput(const blueboat_navigation::DubinInput& msg)
  {
    steering::State start;
    start.x     = msg.start.pose.position.x;
    start.y     = msg.start.pose.position.y;
    start.theta = tf2::getYaw(msg.start.pose.orientation);
    start.kappa = 0.0;
    start.d     = 0.0;

    steering::State goal;
    goal.x     = msg.end.pose.position.x;
    goal.y     = msg.end.pose.position.y;
    goal.theta = tf2::getYaw(msg.end.pose.orientation);
    goal.kappa = 0.0;
    goal.d     = 0.0;

    std::vector<steering::State> states = m_stateSpace->get_path(start, goal);

    if (states.empty())
    {
      ROS_WARN("[dubins_planner] empty path returned");
      return;
    }

    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = msg.start.header.frame_id.empty()
                             ? std::string("map")
                             : msg.start.header.frame_id;

    path.poses.reserve(states.size());
    for (const auto& s : states)
    {
      geometry_msgs::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = s.x;
      p.pose.position.y = s.y;
      // Driving direction encoded in z (planar boat, z is otherwise unused).
      // Dubins is forwards-only, so this should be +1 for all states.
      p.pose.position.z = s.d;

      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, s.theta);
      p.pose.orientation.x = q.x();
      p.pose.orientation.y = q.y();
      p.pose.orientation.z = q.z();
      p.pose.orientation.w = q.w();

      path.poses.push_back(p);
    }

    m_pathPub.publish(path);
    ROS_INFO("[dubins_planner] published path with %zu poses", path.poses.size());
  }

  ros::Publisher  m_pathPub;
  ros::Subscriber m_inputSub;
  std::shared_ptr<steering::Dubins_State_Space> m_stateSpace;

  double m_kappaMax;
  double m_discretization;
  bool   m_forwards;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dubins_planner");
  blueboat_navigation::DubinsPlanner node;
  ros::spin();
  return 0;
}