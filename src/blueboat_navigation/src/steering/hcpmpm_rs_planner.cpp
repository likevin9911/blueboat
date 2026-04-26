/**
 * hc_rs_planner.cpp
 *
 * Steering function: HC-Reeds-Shepp using HCpmpm_Reeds_Shepp_State_Space
 * from hbanzhaf/steering_functions. G2 continuous, bidirectional, accepts
 * arbitrary start/goal curvature (kappa) — required for chaining segments
 * with a sampling-based planner like RRT* or a graph search like A*, where
 * each node carries its own curvature.
 *
 * The "pmpm" variant evaluates both positive and negative max curvature at
 * the start and at the goal and returns the shortest path among all four
 * combinations.
 *
 * I/O contract (shared by all four planner nodes in this package):
 *   Subscribes: ~planning_input  (blueboat_navigation/DubinInput)
 *   Publishes:  ~planned_path    (nav_msgs/Path)
 *   Publishes:  ~planned_curvatures (std_msgs/Float32MultiArray)
 *
 * Per-pose data embedded into nav_msgs/Path:
 *   pose.position.x      = x         (planar)
 *   pose.position.y      = y         (planar)
 *   pose.position.z      = state.d   (driving direction: +1, 0, -1)
 *   pose.orientation     = yaw from state.theta (quaternion)
 *
 * State.kappa (signed curvature along the path) is published in parallel
 * on ~planned_curvatures, indexed identically to path.poses. Guidance can
 * subscribe to this to size lookahead by curvature instead of by a yaw
 * heuristic.
 *
 * Params (private namespace):
 *   ~turning_radius   [m]    -> kappa_max = 1 / turning_radius        (default 1.0)
 *   ~max_sharpness    [1/m^2]-> sigma_max, max linear change of curvature (default = kappa_max)
 *   ~path_resolution  [m]    -> discretization between integrated states  (default 0.05)
 *   ~start_kappa      [1/m]  -> override start curvature; NaN => 0     (default NaN)
 *   ~goal_kappa       [1/m]  -> override goal  curvature; NaN => 0     (default NaN)
 *
 * Note: HCpmpm is bidirectional by construction; there is no
 * "forwards_only" parameter.
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/utils.h>

#include <blueboat_navigation/DubinInput.h>

#include "steering_functions/hc_cc_state_space/hcpmpm_reeds_shepp_state_space.hpp"

#include <cmath>
#include <limits>
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

    m_startKappaOverride = nhP.param("start_kappa",
                                     std::numeric_limits<double>::quiet_NaN());
    m_goalKappaOverride  = nhP.param("goal_kappa",
                                     std::numeric_limits<double>::quiet_NaN());

    m_stateSpace = std::make_shared<steering::HCpmpm_Reeds_Shepp_State_Space>(
        m_kappaMax, m_sigmaMax, m_discretization);

    m_pathPub      = nhP.advertise<nav_msgs::Path>("planned_path", 10);
    m_curvaturePub = nhP.advertise<std_msgs::Float32MultiArray>(
        "planned_curvatures", 10);

    m_inputSub = nhP.subscribe("planning_input", 10,
                               &HCReedsSheppPlanner::onInput, this);

    ROS_INFO("[hcpmpm_rs_planner] HCpmpm  kappa_max=%.3f (R=%.2f m)  sigma=%.3f  res=%.3f m  bidirectional",
             m_kappaMax, turning_radius, m_sigmaMax, m_discretization);
  }

private:
  void onInput(const blueboat_navigation::DubinInput& msg)
  {
    steering::State start;
    start.x     = msg.start.pose.position.x;
    start.y     = msg.start.pose.position.y;
    start.theta = tf2::getYaw(msg.start.pose.orientation);
    start.kappa = std::isnan(m_startKappaOverride) ? 0.0 : m_startKappaOverride;
    start.d     = 0.0;

    steering::State goal;
    goal.x     = msg.end.pose.position.x;
    goal.y     = msg.end.pose.position.y;
    goal.theta = tf2::getYaw(msg.end.pose.orientation);
    goal.kappa = std::isnan(m_goalKappaOverride) ? 0.0 : m_goalKappaOverride;
    goal.d     = 0.0;

    std::vector<steering::State> states = m_stateSpace->get_path(start, goal);

    if (states.empty())
    {
      ROS_WARN("[hcpmpm_rs_planner] empty path returned");
      return;
    }

    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = msg.start.header.frame_id.empty()
                             ? std::string("map")
                             : msg.start.header.frame_id;

    std_msgs::Float32MultiArray curvatures;
    curvatures.data.reserve(states.size());

    path.poses.reserve(states.size());
    for (const auto& s : states)
    {
      geometry_msgs::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = s.x;
      p.pose.position.y = s.y;
      // Driving direction from the steering library:
      //   +1 = forward, -1 = reverse, 0 = transition (cusp).
      p.pose.position.z = s.d;

      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, s.theta);
      p.pose.orientation.x = q.x();
      p.pose.orientation.y = q.y();
      p.pose.orientation.z = q.z();
      p.pose.orientation.w = q.w();

      path.poses.push_back(p);
      curvatures.data.push_back(static_cast<float>(s.kappa));
    }

    m_pathPub.publish(path);
    m_curvaturePub.publish(curvatures);
    ROS_INFO("[hcpmpm_rs_planner] published path with %zu poses", path.poses.size());
  }

  ros::Publisher  m_pathPub;
  ros::Publisher  m_curvaturePub;
  ros::Subscriber m_inputSub;
  std::shared_ptr<steering::HCpmpm_Reeds_Shepp_State_Space> m_stateSpace;

  double m_kappaMax;
  double m_sigmaMax;
  double m_discretization;
  double m_startKappaOverride;
  double m_goalKappaOverride;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hcpmpm_rs_planner");
  blueboat_navigation::HCReedsSheppPlanner node;
  ros::spin();
  return 0;
}