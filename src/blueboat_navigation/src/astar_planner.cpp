/**
 * a_star_planner.cpp
 *
 * Global search over an OccupancyGrid (typically /inflated_map). Reads
 * /global_goal, looks up the robot's pose via tf, runs 8-connected A*
 * on a downsampled grid, smooths the result with line-of-sight reduction
 * (Bresenham), and publishes a sparse list of waypoints.
 *
 * Subscribes:
 *   ~map     (nav_msgs/OccupancyGrid, default remap from /inflated_map)
 *   ~goal    (geometry_msgs/PoseStamped, default remap from /global_goal)
 *
 * Publishes:
 *   ~waypoints (nav_msgs/Path) — sparse, post-LoS-smoothing
 *
 * Params (private):
 *   ~planner_resolution  [m]     downsampled grid cell size for search.
 *                                 0.5 m is a good default; finer is wasted
 *                                 work since HC-RS will smooth between
 *                                 consecutive waypoints anyway. Default 0.5.
 *   ~obstacle_threshold  [int]   occupancy values strictly greater than
 *                                 this are treated as obstacles. Default 50.
 *   ~allow_unknown       [bool]  treat unknown (-1) cells as free. Default true
 *                                 (the inflated map will have some unknown
 *                                 around frontiers).
 *   ~global_frame        [str]   frame the goal/path are expressed in.
 *                                 Default "map".
 *   ~robot_frame         [str]   tf frame for robot pose. Default "base_link".
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
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <utility>

namespace blueboat_navigation
{

class AStarPlanner
{
public:
  AStarPlanner()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    m_plannerResolution = nhP.param("planner_resolution", 0.5);
    m_obstacleThreshold = nhP.param("obstacle_threshold", 50);
    m_allowUnknown      = nhP.param("allow_unknown", true);
    nhP.param<std::string>("global_frame", m_globalFrame, "map");
    nhP.param<std::string>("robot_frame",  m_robotFrame,  "base_link");

    m_mapSub  = nhP.subscribe("map",  1, &AStarPlanner::mapCallback,  this);
    m_goalSub = nhP.subscribe("goal", 1, &AStarPlanner::goalCallback, this);

    m_pathPub = nhP.advertise<nav_msgs::Path>("waypoints", 1, true);

    m_tfListener = std::make_shared<tf2_ros::TransformListener>(m_tfBuffer);

    ROS_INFO("[a_star_planner] resolution=%.2f m  obstacle_thresh=%d  unknown_free=%s",
             m_plannerResolution, m_obstacleThreshold,
             m_allowUnknown ? "true" : "false");
  }

private:
  // -------- callbacks --------
  void mapCallback(const nav_msgs::OccupancyGrid& msg) { m_map = msg; m_haveMap = true; }

  void goalCallback(const geometry_msgs::PoseStamped& goal)
  {
    if (!m_haveMap)
    {
      ROS_WARN("[a_star_planner] no map yet; ignoring goal");
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
      ROS_WARN("[a_star_planner] tf lookup failed: %s", ex.what());
      return;
    }

    double sx = tf.transform.translation.x;
    double sy = tf.transform.translation.y;
    double gx = goal.pose.position.x;
    double gy = goal.pose.position.y;

    // Convert world to planner-grid indices
    int sgx, sgy, ggx, ggy;
    if (!worldToGrid(sx, sy, sgx, sgy) || !worldToGrid(gx, gy, ggx, ggy))
    {
      ROS_WARN("[a_star_planner] start or goal outside map bounds");
      return;
    }

    if (isObstacle(ggx, ggy))
    {
      ROS_WARN("[a_star_planner] goal cell is obstacle/unknown");
      return;
    }

    auto rawCells = aStar({sgx, sgy}, {ggx, ggy});
    if (rawCells.empty())
    {
      ROS_WARN("[a_star_planner] A* found no path");
      return;
    }

    auto smoothed = lineOfSightSmooth(rawCells);

    // Build Path: orientations point along the line from each waypoint to
    // the next so HC-RS gets a sensible heading at every node.
    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = m_globalFrame;
    path.poses.reserve(smoothed.size());

    for (size_t i = 0; i < smoothed.size(); ++i)
    {
      double wx, wy;
      gridToWorld(smoothed[i].first, smoothed[i].second, wx, wy);

      double yaw;
      if (i + 1 < smoothed.size())
      {
        double nwx, nwy;
        gridToWorld(smoothed[i + 1].first, smoothed[i + 1].second, nwx, nwy);
        yaw = std::atan2(nwy - wy, nwx - wx);
      }
      else
      {
        yaw = tf2::getYaw(goal.pose.orientation);
      }

      // Override start orientation with current robot heading so the first
      // HC-RS segment plans from the boat's actual pose.
      if (i == 0)
      {
        yaw = tf2::getYaw(tf.transform.rotation);
      }

      geometry_msgs::PoseStamped p;
      p.header = path.header;
      p.pose.position.x = wx;
      p.pose.position.y = wy;
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
    ROS_INFO("[a_star_planner] published %zu waypoints (raw %zu)",
             path.poses.size(), rawCells.size());
  }

  // -------- grid helpers --------
  // Planner grid is the inflated map downsampled to m_plannerResolution.
  // We don't actually rebuild a grid; we sample the underlying map at
  // planner-grid cell centers when checking occupancy.

  bool worldToGrid(double wx, double wy, int& gx, int& gy) const
  {
    double ox = m_map.info.origin.position.x;
    double oy = m_map.info.origin.position.y;
    double mapW = m_map.info.width  * m_map.info.resolution;
    double mapH = m_map.info.height * m_map.info.resolution;

    if (wx < ox || wy < oy || wx > ox + mapW || wy > oy + mapH)
      return false;

    gx = static_cast<int>(std::floor((wx - ox) / m_plannerResolution));
    gy = static_cast<int>(std::floor((wy - oy) / m_plannerResolution));
    return true;
  }

  void gridToWorld(int gx, int gy, double& wx, double& wy) const
  {
    double ox = m_map.info.origin.position.x;
    double oy = m_map.info.origin.position.y;
    wx = ox + (gx + 0.5) * m_plannerResolution;
    wy = oy + (gy + 0.5) * m_plannerResolution;
  }

  // Returns true if the underlying map shows any obstacle in the cell.
  // Conservative: if ANY underlying map cell inside the planner cell is
  // occupied, the planner cell is occupied.
  bool isObstacle(int gx, int gy) const
  {
    double mapRes = m_map.info.resolution;
    if (mapRes <= 0.0) return true;

    double ox = m_map.info.origin.position.x;
    double oy = m_map.info.origin.position.y;
    double cellMinX = ox + gx * m_plannerResolution;
    double cellMinY = oy + gy * m_plannerResolution;

    int mxLo = static_cast<int>(std::floor((cellMinX - ox) / mapRes));
    int myLo = static_cast<int>(std::floor((cellMinY - oy) / mapRes));
    int mxHi = static_cast<int>(std::ceil ((cellMinX + m_plannerResolution - ox) / mapRes));
    int myHi = static_cast<int>(std::ceil ((cellMinY + m_plannerResolution - oy) / mapRes));

    int W = static_cast<int>(m_map.info.width);
    int H = static_cast<int>(m_map.info.height);

    if (mxHi <= 0 || myHi <= 0 || mxLo >= W || myLo >= H)
      return true;

    mxLo = std::max(0, mxLo);
    myLo = std::max(0, myLo);
    mxHi = std::min(W, mxHi);
    myHi = std::min(H, myHi);

    for (int my = myLo; my < myHi; ++my)
    {
      for (int mx = mxLo; mx < mxHi; ++mx)
      {
        int idx = my * W + mx;
        int8_t v = m_map.data[idx];
        if (v > m_obstacleThreshold) return true;
        if (v < 0 && !m_allowUnknown) return true;
      }
    }
    return false;
  }

  // -------- A* --------
  using Cell = std::pair<int, int>;

  struct Node
  {
    Cell pos;
    Cell parent;
    double g, f;
  };

  struct NodeCmp
  {
    bool operator()(const Node& a, const Node& b) const { return a.f > b.f; }
  };

  static double octile(const Cell& a, const Cell& b)
  {
    double dx = std::abs(a.first  - b.first);
    double dy = std::abs(a.second - b.second);
    return (dx + dy) + (std::sqrt(2.0) - 2.0) * std::min(dx, dy);
  }

  std::vector<Cell> aStar(const Cell& start, const Cell& goal)
  {
    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    std::map<Cell, double> bestG;
    std::map<Cell, Cell>   parent;

    Node s{start, start, 0.0, octile(start, goal)};
    open.push(s);
    bestG[start] = 0.0;

    static const int dxs[8] = {1, -1, 0,  0, 1,  1, -1, -1};
    static const int dys[8] = {0,  0, 1, -1, 1, -1,  1, -1};

    while (!open.empty())
    {
      Node q = open.top();
      open.pop();

      auto it = bestG.find(q.pos);
      if (it != bestG.end() && q.g > it->second + 1e-9) continue;

      if (q.pos == goal)
      {
        std::vector<Cell> rev;
        Cell c = goal;
        rev.push_back(c);
        while (c != start)
        {
          c = parent[c];
          rev.push_back(c);
        }
        std::reverse(rev.begin(), rev.end());
        return rev;
      }

      for (int k = 0; k < 8; ++k)
      {
        Cell n{q.pos.first + dxs[k], q.pos.second + dys[k]};
        if (isObstacle(n.first, n.second)) continue;

        // Diagonal corner-cutting check: refuse diagonals if either
        // orthogonal neighbor is blocked.
        if (k >= 4)
        {
          if (isObstacle(q.pos.first + dxs[k], q.pos.second)) continue;
          if (isObstacle(q.pos.first, q.pos.second + dys[k])) continue;
        }

        double step = (k < 4) ? 1.0 : std::sqrt(2.0);
        double ng = q.g + step;

        auto it2 = bestG.find(n);
        if (it2 == bestG.end() || ng < it2->second - 1e-9)
        {
          bestG[n] = ng;
          parent[n] = q.pos;
          Node next{n, q.pos, ng, ng + octile(n, goal)};
          open.push(next);
        }
      }
    }

    return {};
  }

  // -------- Bresenham line-of-sight --------
  bool losClear(const Cell& a, const Cell& b) const
  {
    int x0 = a.first,  y0 = a.second;
    int x1 = b.first,  y1 = b.second;
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    while (true)
    {
      if (isObstacle(x0, y0)) return false;
      if (x0 == x1 && y0 == y1) return true;
      int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  // Greedy LoS smoothing: from each kept point, jump as far ahead as
  // visibility allows. Same idea as your aStarSPT post-process.
  std::vector<Cell> lineOfSightSmooth(const std::vector<Cell>& raw)
  {
    if (raw.size() <= 2) return raw;
    std::vector<Cell> out;
    out.push_back(raw.front());
    size_t anchor = 0;
    for (size_t probe = 2; probe < raw.size(); ++probe)
    {
      if (!losClear(raw[anchor], raw[probe]))
      {
        out.push_back(raw[probe - 1]);
        anchor = probe - 1;
      }
    }
    out.push_back(raw.back());
    return out;
  }

  // -------- members --------
  ros::Subscriber m_mapSub;
  ros::Subscriber m_goalSub;
  ros::Publisher  m_pathPub;

  tf2_ros::Buffer m_tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tfListener;

  nav_msgs::OccupancyGrid m_map;
  bool m_haveMap{false};

  double m_plannerResolution;
  int    m_obstacleThreshold;
  bool   m_allowUnknown;
  std::string m_globalFrame;
  std::string m_robotFrame;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "a_star_planner");
  blueboat_navigation::AStarPlanner node;
  ros::spin();
  return 0;
}