/**
 * cc_dubins_planner.cpp
 *
 * Steering function: CC-Dubins (Continuous Curvature Dubins, G2 continuous,
 * forwards-only). Uses CC00_Dubins_State_Space from hbanzhaf/steering_functions
 * with zero curvature at start and goal.
 *
 * I/O contract (shared by all four planner nodes in this package):
 *   Subscribes: ~planning_input  (blueboat_navigation/DubinInput)
 *   Publishes:  ~planned_path    (nav_msgs/Path)
 *
 * Driving direction is taken directly from steering::State::d and embedded
 * in pose.position.z (see dubins_planner.cpp for rationale). For CC-Dubins
 * this is always +1 (forwards).
 *
 * Params (private namespace):
 *   ~turning_radius   [m]    -> kappa_max = 1 / turning_radius        (default 1.0)
 *   ~max_sharpness    [1/m^2]-> sigma_max, max linear change of curvature (default = kappa_max)
 *   ~path_resolution  [m]    -> discretization between integrated states  (default 0.05)
 *   ~forwards_only    [bool] -> exposed for completeness; CC-Dubins is forwards by definition (default true)
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#include <blueboat_navigation/DubinInput.h>

#include "steering_functions/hc_cc_state_space/cc00_dubins_state_space.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace blueboat_navigation
{

class CCDubinsPlanner
{
public:
  CCDubinsPlanner()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    double turning_radius = nhP.param("turning_radius", 1.0);
    m_kappaMax       = 1.0 / turning_radius;
    // Default sigma to kappa (path-length-style choice for "smoothly curling
    // up to max curvature in ~1 m"). User can override.
    m_sigmaMax       = nhP.param("max_sharpness", m_kappaMax);
    m_discretization = nhP.param("path_resolution", 0.05);
    m_forwards       = nhP.param("forwards_only", true);

    m_stateSpace = std::make_shared<steering::CC00_Dubins_State_Space>(
        m_kappaMax, m_sigmaMax, m_discretization, m_forwards);

    m_pathPub  = nhP.advertise<nav_msgs::Path>("planned_path", 10);
    m_inputSub = nhP.subscribe("planning_input", 10,
                               &CCDubinsPlanner::onInput, this);

    ROS_INFO("[cc_dubins_planner] kappa_max=%.3f (R=%.2f m)  sigma=%.3f  res=%.3f m  forwards=%s",
             m_kappaMax, turning_radius, m_sigmaMax, m_discretization,
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
      ROS_WARN("[cc_dubins_planner] empty path returned");
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
      // Driving direction (always +1 here; kept for symmetry with RS planners).
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
    ROS_INFO("[cc_dubins_planner] published path with %zu poses", path.poses.size());
  }

  ros::Publisher  m_pathPub;
  ros::Subscriber m_inputSub;
  std::shared_ptr<steering::CC00_Dubins_State_Space> m_stateSpace;

  double m_kappaMax;
  double m_sigmaMax;
  double m_discretization;
  bool   m_forwards;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cc_dubins_planner");
  blueboat_navigation::CCDubinsPlanner node;
  ros::spin();
  return 0;
}