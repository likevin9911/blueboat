/**
 * hc_rs_planner.cpp HC00
 *
 * Steering function: HC-Reeds-Shepp (Hybrid Curvature Reeds-Shepp,
 * G2 continuous, bidirectional). Uses HC00_Reeds_Shepp_State_Space from
 * hbanzhaf/steering_functions with zero curvature at start and goal.
 *
 * I/O contract (shared by all four planner nodes in this package):
 *   Subscribes: ~planning_input  (blueboat_navigation/DubinInput)
 *   Publishes:  ~planned_path    (nav_msgs/Path)
 *
 * Driving direction is taken directly from steering::State::d ∈ {-1, 0, +1}
 * and embedded in pose.position.z, identical to the RS planner. Guidance
 * reads pose.position.z verbatim instead of inferring direction.
 *
 * Params (private namespace):
 *   ~turning_radius   [m]    -> kappa_max = 1 / turning_radius        (default 1.0)
 *   ~max_sharpness    [1/m^2]-> sigma_max, max linear change of curvature (default = kappa_max)
 *   ~path_resolution  [m]    -> discretization between integrated states  (default 0.05)
 *
 * Note: HC-RS is bidirectional by construction; there is no "forwards_only"
 * parameter.
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#include <blueboat_navigation/DubinInput.h>

#include "steering_functions/hc_cc_state_space/hc00_reeds_shepp_state_space.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace blueboat_navigation
{

class HCReedsSheppPlanner
{
public:
  HCReedsSheppPlanner()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    double turning_radius = nhP.param("turning_radius", 1.0);
    m_kappaMax       = 1.0 / turning_radius;
    m_sigmaMax       = nhP.param("max_sharpness", m_kappaMax);
    m_discretization = nhP.param("path_resolution", 0.05);

    m_stateSpace = std::make_shared<steering::HC00_Reeds_Shepp_State_Space>(
        m_kappaMax, m_sigmaMax, m_discretization);

    m_pathPub  = nh.advertise<nav_msgs::Path>("planned_path", 10);
    m_inputSub = nhP.subscribe("planning_input", 10,
                               &HCReedsSheppPlanner::onInput, this);

    ROS_INFO("[hc_rs_planner] kappa_max=%.3f (R=%.2f m)  sigma=%.3f  res=%.3f m  bidirectional",
             m_kappaMax, turning_radius, m_sigmaMax, m_discretization);
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
      ROS_WARN("[hc_rs_planner] empty path returned");
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
      // Driving direction from steering library:
      //   +1 = forward, -1 = reverse, 0 = transition (cusp).
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
    ROS_INFO("[hc_rs_planner] published path with %zu poses", path.poses.size());
  }

  ros::Publisher  m_pathPub;
  ros::Subscriber m_inputSub;
  std::shared_ptr<steering::HC00_Reeds_Shepp_State_Space> m_stateSpace;

  double m_kappaMax;
  double m_sigmaMax;
  double m_discretization;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hc_rs_planner");
  blueboat_navigation::HCReedsSheppPlanner node;
  ros::spin();
  return 0;
}