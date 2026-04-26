/**
 * rrt_star_planner.cpp
 *
 * Geometric RRT* over an OccupancyGrid (typically /inflated_map). Reads
 * /global_goal, looks up the robot's pose via tf, samples a bounded box
 * around (start, goal), runs RRT* with a Euclidean metric and straight-
 * line collision checking, and publishes a sparse list of waypoints.
 *
 * Why geometric RRT* and not kinodynamic? The downstream HC-RS planner
 * (HCpmpm_Reeds_Shepp_State_Space) provides nonholonomic-feasible
 * steering between consecutive waypoints. Doing kinodynamic RRT* with
 * HC-RS as the steering function is correct but slow; for a USV in
 * mostly open water, geometric RRT* + HC-RS post-process gives most of
 * the benefit at a fraction of the cost. If you need true kinodynamic
 * cost-optimality, swap the Euclidean metric here for HC-RS distance.
 *
 * Subscribes:
 *   ~map     (nav_msgs/OccupancyGrid, default remap from /inflated_map)
 *   ~goal    (geometry_msgs/PoseStamped, default remap from /global_goal)
 *
 * Publishes:
 *   ~waypoints (nav_msgs/Path) — sparse waypoint list
 *
 * Params (private):
 *   ~max_iterations     [int]  default 5000
 *   ~goal_bias          [0..1] probability of sampling the goal directly. default 0.05
 *   ~step_size          [m]    max edge length for the steer step. default 2.0
 *   ~rewire_radius      [m]    radius used in RRT* rewiring. default 4.0
 *   ~goal_tolerance     [m]    distance below which a node counts as having reached the goal. default 1.0
 *   ~pad_margin         [m]    sampling box padding around start+goal bbox. default 10.0
 *   ~obstacle_threshold [int]  occupancy values strictly greater than this are obstacles. default 50
 *   ~allow_unknown      [bool] treat -1 cells as free. default true
 *   ~collision_step     [m]    edge collision-check step size. default 0.2
 *   ~global_frame       [str]  default "map"
 *   ~robot_frame        [str]  default "base_link"
 */

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace blueboat_navigation
{

class RRTStarPlanner
{
public:
  RRTStarPlanner() : m_rng(std::random_device{}())
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    m_maxIters         = nhP.param("max_iterations", 5000);
    m_goalBias         = nhP.param("goal_bias", 0.05);
    m_stepSize         = nhP.param("step_size", 2.0);
    m_rewireRadius     = nhP.param("rewire_radius", 4.0);
    m_goalTolerance    = nhP.param("goal_tolerance", 1.0);
    m_padMargin        = nhP.param("pad_margin", 10.0);
    m_obstacleThreshold= nhP.param("obstacle_threshold", 50);
    m_allowUnknown     = nhP.param("allow_unknown", true);
    m_collisionStep    = nhP.param("collision_step", 0.2);
    nhP.param<std::string>("global_frame", m_globalFrame, "map");
    nhP.param<std::string>("robot_frame",  m_robotFrame,  "base_link");

    m_mapSub  = nhP.subscribe("map",  1, &RRTStarPlanner::mapCallback,  this);
    m_goalSub = nhP.subscribe("goal", 1, &RRTStarPlanner::goalCallback, this);

    m_pathPub = nhP.advertise<nav_msgs::Path>("waypoints", 1, true);

    m_tfListener = std::make_shared<tf2_ros::TransformListener>(m_tfBuffer);

    ROS_INFO("[rrt_star_planner] iters=%d  step=%.2f m  rewire=%.2f m  bias=%.2f  pad=%.1f m",
             m_maxIters, m_stepSize, m_rewireRadius, m_goalBias, m_padMargin);
  }

private:
  // -------- callbacks --------
  void mapCallback(const nav_msgs::OccupancyGrid& msg) { m_map = msg; m_haveMap = true; }

  void goalCallback(const geometry_msgs::PoseStamped& goal)
  {
    if (!m_haveMap)
    {
      ROS_WARN("[rrt_star_planner] no map yet; ignoring goal");
      return;
    }

    geometry_msgs::TransformStamped tf;
    try
    {
      tf = m_tfBuffer.lookupTransform(m_globalFrame, m_robotFrame,
                                       ros::Time(0), ros::Duration(1.0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN("[rrt_star_planner] tf lookup failed: %s", ex.what());
      return;
    }

    double sx = tf.transform.translation.x;
    double sy = tf.transform.translation.y;
    double startYaw = tf2::getYaw(tf.transform.rotation);
    double gx = goal.pose.position.x;
    double gy = goal.pose.position.y;
    double goalYaw = tf2::getYaw(goal.pose.orientation);

    if (inObstacle(gx, gy))
    {
      ROS_WARN("[rrt_star_planner] goal is in obstacle/unknown");
      return;
    }

    // Sampling box: bbox of (start, goal) padded by pad_margin, clipped to map.
    double mox = m_map.info.origin.position.x;
    double moy = m_map.info.origin.position.y;
    double mw  = m_map.info.width  * m_map.info.resolution;
    double mh  = m_map.info.height * m_map.info.resolution;

    double xMin = std::max(mox,         std::min(sx, gx) - m_padMargin);
    double xMax = std::min(mox + mw,    std::max(sx, gx) + m_padMargin);
    double yMin = std::max(moy,         std::min(sy, gy) - m_padMargin);
    double yMax = std::min(moy + mh,    std::max(sy, gy) + m_padMargin);

    auto rawPath = runRRTStar(sx, sy, gx, gy, xMin, xMax, yMin, yMax);
    if (rawPath.empty())
    {
      ROS_WARN("[rrt_star_planner] no path found within %d iterations", m_maxIters);
      return;
    }

    // Build the output Path. Heading at each node = atan2 to next node;
    // start uses robot heading; goal uses goal orientation.
    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = m_globalFrame;
    path.poses.reserve(rawPath.size());

    for (size_t i = 0; i < rawPath.size(); ++i)
    {
      double yaw;
      if (i == 0)                       yaw = startYaw;
      else if (i + 1 == rawPath.size()) yaw = goalYaw;
      else
      {
        yaw = std::atan2(rawPath[i + 1].second - rawPath[i].second,
                         rawPath[i + 1].first  - rawPath[i].first);
      }
      geometry_msgs::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = rawPath[i].first;
      p.pose.position.y = rawPath[i].second;
      p.pose.position.z = 0.0;
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);
      p.pose.orientation.x = q.x();
      p.pose.orientation.y = q.y();
      p.pose.orientation.z = q.z();
      p.pose.orientation.w = q.w();
      path.poses.push_back(p);
    }

    m_pathPub.publish(path);
    ROS_INFO("[rrt_star_planner] published path with %zu waypoints", path.poses.size());
  }

  // -------- collision checks --------
  bool inObstacle(double wx, double wy) const
  {
    double mox = m_map.info.origin.position.x;
    double moy = m_map.info.origin.position.y;
    double mr  = m_map.info.resolution;
    int W = m_map.info.width, H = m_map.info.height;

    int mx = static_cast<int>(std::floor((wx - mox) / mr));
    int my = static_cast<int>(std::floor((wy - moy) / mr));
    if (mx < 0 || my < 0 || mx >= W || my >= H) return true;
    int8_t v = m_map.data[my * W + mx];
    if (v > m_obstacleThreshold) return true;
    if (v < 0 && !m_allowUnknown) return true;
    return false;
  }

  bool edgeClear(double x0, double y0, double x1, double y1) const
  {
    double d = std::hypot(x1 - x0, y1 - y0);
    if (d < 1e-9) return !inObstacle(x0, y0);
    int n = static_cast<int>(std::ceil(d / m_collisionStep));
    for (int i = 0; i <= n; ++i)
    {
      double t = static_cast<double>(i) / n;
      if (inObstacle(x0 + t * (x1 - x0), y0 + t * (y1 - y0))) return false;
    }
    return true;
  }

  // -------- RRT* core --------
  struct TreeNode
  {
    double x, y;
    int    parent;   // index into m_tree; -1 for root
    double cost;
  };

  std::vector<std::pair<double, double>>
  runRRTStar(double sx, double sy, double gx, double gy,
             double xMin, double xMax, double yMin, double yMax)
  {
    std::vector<TreeNode> tree;
    tree.push_back({sx, sy, -1, 0.0});

    std::uniform_real_distribution<double> ux(xMin, xMax);
    std::uniform_real_distribution<double> uy(yMin, yMax);
    std::uniform_real_distribution<double> ub(0.0, 1.0);

    int bestGoalIdx = -1;
    double bestGoalCost = std::numeric_limits<double>::max();

    for (int it = 0; it < m_maxIters; ++it)
    {
      // 1) Sample
      double rx, ry;
      if (ub(m_rng) < m_goalBias) { rx = gx; ry = gy; }
      else { rx = ux(m_rng); ry = uy(m_rng); }

      // 2) Nearest neighbor (linear scan; fine up to ~10k nodes)
      int nearestIdx = 0;
      double nearestD2 = std::numeric_limits<double>::max();
      for (size_t i = 0; i < tree.size(); ++i)
      {
        double dx = tree[i].x - rx, dy = tree[i].y - ry;
        double d2 = dx * dx + dy * dy;
        if (d2 < nearestD2) { nearestD2 = d2; nearestIdx = static_cast<int>(i); }
      }

      // 3) Steer (clamp to step_size)
      double dx = rx - tree[nearestIdx].x;
      double dy = ry - tree[nearestIdx].y;
      double d  = std::hypot(dx, dy);
      double nx, ny;
      if (d <= m_stepSize) { nx = rx; ny = ry; }
      else
      {
        nx = tree[nearestIdx].x + dx / d * m_stepSize;
        ny = tree[nearestIdx].y + dy / d * m_stepSize;
      }

      // 4) Collision check on the candidate edge
      if (!edgeClear(tree[nearestIdx].x, tree[nearestIdx].y, nx, ny)) continue;

      // 5) Find neighbors within rewire radius
      std::vector<int> neighbors;
      for (size_t i = 0; i < tree.size(); ++i)
      {
        double ddx = tree[i].x - nx, ddy = tree[i].y - ny;
        if (ddx * ddx + ddy * ddy <= m_rewireRadius * m_rewireRadius)
          neighbors.push_back(static_cast<int>(i));
      }

      // 6) Choose best parent
      int bestParent = nearestIdx;
      double bestCost = tree[nearestIdx].cost +
                        std::hypot(nx - tree[nearestIdx].x,
                                   ny - tree[nearestIdx].y);
      for (int ni : neighbors)
      {
        double c = tree[ni].cost +
                   std::hypot(nx - tree[ni].x, ny - tree[ni].y);
        if (c < bestCost &&
            edgeClear(tree[ni].x, tree[ni].y, nx, ny))
        {
          bestCost = c;
          bestParent = ni;
        }
      }

      tree.push_back({nx, ny, bestParent, bestCost});
      int newIdx = static_cast<int>(tree.size()) - 1;

      // 7) Rewire neighbors through new node if cheaper
      for (int ni : neighbors)
      {
        if (ni == bestParent) continue;
        double c = bestCost + std::hypot(tree[ni].x - nx, tree[ni].y - ny);
        if (c < tree[ni].cost &&
            edgeClear(nx, ny, tree[ni].x, tree[ni].y))
        {
          tree[ni].parent = newIdx;
          tree[ni].cost   = c;
        }
      }

      // 8) Goal check
      double dgx = nx - gx, dgy = ny - gy;
      if (std::hypot(dgx, dgy) <= m_goalTolerance)
      {
        if (bestCost < bestGoalCost)
        {
          bestGoalCost = bestCost;
          bestGoalIdx  = newIdx;
        }
      }
    }

    if (bestGoalIdx < 0) return {};

    // Trace back, then append exact goal so the final pose is correct.
    std::vector<std::pair<double, double>> path;
    for (int i = bestGoalIdx; i >= 0; i = tree[i].parent)
      path.emplace_back(tree[i].x, tree[i].y);
    std::reverse(path.begin(), path.end());

    // Replace the final node with the actual goal coords if within tolerance.
    if (!path.empty()) path.back() = {gx, gy};
    return path;
  }

  // -------- members --------
  ros::Subscriber m_mapSub;
  ros::Subscriber m_goalSub;
  ros::Publisher  m_pathPub;

  tf2_ros::Buffer m_tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tfListener;

  nav_msgs::OccupancyGrid m_map;
  bool m_haveMap{false};

  int    m_maxIters;
  double m_goalBias;
  double m_stepSize;
  double m_rewireRadius;
  double m_goalTolerance;
  double m_padMargin;
  int    m_obstacleThreshold;
  bool   m_allowUnknown;
  double m_collisionStep;
  std::string m_globalFrame;
  std::string m_robotFrame;

  std::mt19937 m_rng;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rrt_star_planner");
  blueboat_navigation::RRTStarPlanner node;
  ros::spin();
  return 0;
}
