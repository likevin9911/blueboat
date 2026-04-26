// dock_approach_planner.cpp
//
// 5-scenario docking with A* path planning.
//
// DOCK FRAME (from L-shape estimator):
//   dock_yaw = along LONG AXIS
//   long_axis  = dock_yaw direction
//   short_axis = dock_yaw + 90°  (points left looking inland)
//
// Parameters:
//   dock_side: "port" or "starboard"
//     port      → dock against LEFT wall  (+short_axis face)
//     starboard → dock against RIGHT wall (-short_axis face)
//
//   bow_direction: "+x", "-x", or "-y"
//     +x → bow along +long_axis
//     -x → bow along -long_axis
//     -y → bow perpendicular away from dock wall
//
// 5 scenarios:
//   1. -x, starboard  — bow down, starboard against right wall
//   2. -x, port       — bow down, port against left wall
//   3. +x, starboard  — bow up, starboard against right wall
//   4. +x, port       — bow up, port against left wall
//   5. -y, stern      — bow away from wall, stern into dock
//
// Goal = wall_edge + inflation + standoff perpendicular to wall.
// Staging placed so the boat approaches goal from a direction
// where the goal is AHEAD (forward reachable).
//
// A* on inflated costmap routes around the dock.
// Path re-published every path_update_interval seconds.

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Bool.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>

static double wrapAngle(double a)
{
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

static geometry_msgs::PoseStamped makePose(const std::string& frame,
                                           double x, double y, double yaw)
{
  geometry_msgs::PoseStamped ps;
  ps.header.frame_id = frame;
  ps.header.stamp    = ros::Time::now();
  ps.pose.position.x = x;
  ps.pose.position.y = y;
  ps.pose.position.z = 0.0;
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  ps.pose.orientation = tf2::toMsg(q);
  return ps;
}

// =========================================================================
// A* on occupancy grid
// =========================================================================
struct AStarNode {
  int x, y;
  double g, f;
  bool operator>(const AStarNode& o) const { return f > o.f; }
};

static std::vector<std::pair<double,double>> astarSearch(
    const nav_msgs::OccupancyGrid& map,
    double wx0, double wy0, double wx1, double wy1,
    int lethal_thresh)
{
  const auto& info = map.info;
  int W = info.width, H = info.height;
  double res = info.resolution;
  double ox = info.origin.position.x;
  double oy = info.origin.position.y;

  auto toGrid = [&](double wx, double wy, int& gx, int& gy) {
    gx = (int)((wx - ox) / res);
    gy = (int)((wy - oy) / res);
  };
  auto toWorld = [&](int gx, int gy) -> std::pair<double,double> {
    return {ox + (gx + 0.5) * res, oy + (gy + 0.5) * res};
  };

  int sx, sy, gx, gy;
  toGrid(wx0, wy0, sx, sy);
  toGrid(wx1, wy1, gx, gy);

  auto valid = [&](int x, int y) { return x >= 0 && y >= 0 && x < W && y < H; };
  auto free = [&](int x, int y) -> bool {
    if (!valid(x, y)) return false;
    int8_t c = map.data[y * W + x];
    return c >= 0 && c < lethal_thresh;
  };

  // Find nearest free cell if start/goal is occupied
  auto findFree = [&](int& cx, int& cy) {
    if (free(cx, cy)) return;
    for (int r = 1; r < 30; ++r)
      for (int dx = -r; dx <= r; ++dx)
        for (int dy = -r; dy <= r; ++dy)
          if (free(cx+dx, cy+dy)) { cx += dx; cy += dy; return; }
  };
  findFree(sx, sy);
  findFree(gx, gy);

  if (!valid(sx, sy) || !valid(gx, gy)) return {};

  int N = W * H;
  std::vector<double> g_cost(N, 1e18);
  std::vector<int> parent(N, -1);
  std::vector<bool> closed(N, false);

  auto idx = [&](int x, int y) { return y * W + x; };
  auto heur = [&](int x, int y) { return std::hypot(x - gx, y - gy); };

  std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open;
  g_cost[idx(sx, sy)] = 0;
  open.push({sx, sy, 0, heur(sx, sy)});

  const int dx8[] = {1,-1,0,0,1,1,-1,-1};
  const int dy8[] = {0,0,1,-1,1,-1,1,-1};
  const double dc[] = {1,1,1,1,1.414,1.414,1.414,1.414};

  bool found = false;
  int iter = 0;
  while (!open.empty() && iter < 300000) {
    iter++;
    AStarNode cur = open.top(); open.pop();
    int ci = idx(cur.x, cur.y);
    if (closed[ci]) continue;
    closed[ci] = true;
    if (cur.x == gx && cur.y == gy) { found = true; break; }
    for (int d = 0; d < 8; ++d) {
      int nx = cur.x + dx8[d], ny = cur.y + dy8[d];
      if (!free(nx, ny)) continue;
      int ni = idx(nx, ny);
      double ng = cur.g + dc[d];
      if (ng < g_cost[ni]) {
        g_cost[ni] = ng;
        parent[ni] = ci;
        open.push({nx, ny, ng, ng + heur(nx, ny)});
      }
    }
  }

  if (!found) return {};

  // Backtrack and convert to world coords
  std::vector<std::pair<double,double>> path;
  int ci = idx(gx, gy);
  while (ci >= 0) {
    int cy = ci / W, cx = ci % W;
    path.push_back(toWorld(cx, cy));
    ci = parent[ci];
  }
  std::reverse(path.begin(), path.end());
  return path;
}

// Convert world-coordinate A* path to nav_msgs::Path with headings
static nav_msgs::Path worldPathToNav(const std::string& frame,
                                     const std::vector<std::pair<double,double>>& pts,
                                     double final_yaw,
                                     double downsample = 1.0)
{
  nav_msgs::Path path;
  path.header.frame_id = frame;
  path.header.stamp = ros::Time::now();
  if (pts.empty()) return path;

  // Downsample
  std::vector<std::pair<double,double>> sparse;
  sparse.push_back(pts.front());
  for (size_t i = 1; i < pts.size(); ++i) {
    double d = std::hypot(pts[i].first - sparse.back().first,
                          pts[i].second - sparse.back().second);
    if (d >= downsample || i == pts.size() - 1)
      sparse.push_back(pts[i]);
  }

  for (size_t i = 0; i < sparse.size(); ++i) {
    double yaw;
    if (i + 1 < sparse.size())
      yaw = std::atan2(sparse[i+1].second - sparse[i].second,
                       sparse[i+1].first - sparse[i].first);
    else
      yaw = final_yaw;
    path.poses.push_back(makePose(frame, sparse[i].first, sparse[i].second, yaw));
  }
  return path;
}

// Straight line fallback
static nav_msgs::Path straightPath(const std::string& frame,
                                   double x0, double y0,
                                   double x1, double y1,
                                   double final_yaw, double res = 0.5)
{
  nav_msgs::Path path;
  path.header.frame_id = frame;
  path.header.stamp = ros::Time::now();
  double dx = x1-x0, dy = y1-y0, dist = std::hypot(dx, dy);
  if (dist < 0.2) { path.poses.push_back(makePose(frame,x1,y1,final_yaw)); return path; }
  double yaw = std::atan2(dy, dx);
  int steps = std::max(2, (int)(dist / res));
  for (int i = 0; i <= steps; ++i) {
    double f = (double)i / steps;
    path.poses.push_back(makePose(frame, x0+f*dx, y0+f*dy,
                                  (i == steps) ? final_yaw : yaw));
  }
  return path;
}

// =========================================================================
class DockApproachPlanner
{
public:
  DockApproachPlanner() : nh_(), pnh_("~"), tf_listener_(tf_buffer_),
                          state_(WAITING), dock_received_(false),
                          dims_received_(false), have_map_(false)
  {
    pnh_.param<std::string>("map_frame",       map_frame_,       "map");
    pnh_.param<double>("start_delay",          start_delay_,     8.0);

    // Which wall: "port" (left/+short) or "starboard" (right/-short)
    pnh_.param<std::string>("dock_side",       dock_side_,       "starboard");

    // Bow direction: "+x", "-x", "-y"
    pnh_.param<std::string>("bow_direction",   bow_direction_,   "+x");

    pnh_.param<double>("berth_standoff",       berth_standoff_,  1.0);
    pnh_.param<double>("inflation_radius",     inflation_radius_, 3.0);
    pnh_.param<double>("berth_long_offset",    berth_long_,      0.0);
    pnh_.param<double>("staging_distance",     staging_dist_,    8.0);
    pnh_.param<double>("staging_tolerance",    staging_tol_,     2.0);
    pnh_.param<double>("goal_tolerance",       goal_tolerance_,  1.2);
    pnh_.param<double>("dock_pose_timeout",    dock_timeout_,    5.0);
    pnh_.param<double>("path_update_interval", path_interval_,   5.0);
    pnh_.param<int>("costmap_lethal_thresh",   lethal_thresh_,   50);
    pnh_.param<double>("path_downsample",      path_downsample_, 1.0);

    dock_sub_ = nh_.subscribe("/dock_pose", 5, &DockApproachPlanner::dockPoseCb, this);
    dims_sub_ = nh_.subscribe("/dock_dims", 5, &DockApproachPlanner::dimsCb, this);
    map_sub_  = nh_.subscribe("/inflated_map", 1, &DockApproachPlanner::mapCb, this);
    path_pub_   = nh_.advertise<nav_msgs::Path>("simple_dubins_path", 5, true);
    docked_pub_ = nh_.advertise<std_msgs::Bool>("docking/docked", 5, true);

    ROS_INFO("[DockApproachPlanner] Ready.");
    ROS_INFO("  dock_side=%s  bow_direction=%s", dock_side_.c_str(), bow_direction_.c_str());
    ROS_INFO("  berth_standoff=%.1f  inflation=%.1f  staging_dist=%.1f",
             berth_standoff_, inflation_radius_, staging_dist_);

    ros::Rate rate(10.0);
    while (nh_.ok()) { update(); ros::spinOnce(); rate.sleep(); }
  }

private:
  enum State { WAITING, APPROACHING, DOCKING, DOCKED };

  void mapCb(const nav_msgs::OccupancyGrid::ConstPtr& m) { map_ = *m; have_map_ = true; }

  void dockPoseCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    geometry_msgs::PoseStamped in_map = *msg;
    if (msg->header.frame_id != map_frame_) {
      try { tf_buffer_.transform(*msg, in_map, map_frame_, ros::Duration(0.1)); }
      catch (tf2::TransformException& ex) { ROS_WARN_THROTTLE(2.0, "TF: %s", ex.what()); return; }
    }
    if (!dock_received_) {
      first_dock_time_ = ros::Time::now();
      dock_received_ = true;
    }
    dock_pose_ = in_map;
    last_dock_time_ = ros::Time::now();
  }

  void dimsCb(const geometry_msgs::Vector3Stamped::ConstPtr& m)
  { dock_length_ = m->vector.x; dock_width_ = m->vector.y; dims_received_ = true; }

  bool getRobotPose(double& rx, double& ry, double& ryaw)
  {
    geometry_msgs::TransformStamped tf;
    try { tf = tf_buffer_.lookupTransform(map_frame_, "base_link", ros::Time(0), ros::Duration(0.0)); }
    catch (tf2::TransformException& ex) { ROS_WARN_THROTTLE(2.0, "TF: %s", ex.what()); return false; }
    rx = tf.transform.translation.x;
    ry = tf.transform.translation.y;
    ryaw = tf2::getYaw(tf.transform.rotation);
    return true;
  }

  // -----------------------------------------------------------------------
  // Compute goal and staging.
  //
  // dock_side determines which wall:
  //   "starboard" → right wall at centroid - half_wid * short_axis
  //                 goal offset further in -short direction
  //   "port"      → left wall at centroid + half_wid * short_axis
  //                 goal offset further in +short direction
  //
  // bow_direction determines heading and staging:
  //   "+x" → bow along +long. Staging behind in -long from goal.
  //   "-x" → bow along -long. Staging behind in +long from goal.
  //   "-y" → bow away from dock wall. Staging on the opposite side
  //          of goal from the dock (further from wall), so the boat
  //          approaches goal heading TOWARD the wall. The A* path
  //          from robot to staging routes around the dock, and the
  //          final leg from staging to goal is a straight approach
  //          toward the dock wall.
  // -----------------------------------------------------------------------
  void computeGoalAndStaging(double& gx, double& gy, double& gyaw,
                             double& sx, double& sy, double& syaw)
  {
    double dock_yaw = tf2::getYaw(dock_pose_.pose.orientation);
    double cx = dock_pose_.pose.position.x;
    double cy = dock_pose_.pose.position.y;

    // Long axis unit vector (along dock_yaw)
    double lx = std::cos(dock_yaw), ly = std::sin(dock_yaw);
    // Short axis unit vector (dock_yaw + 90°, points "left")
    double shx = -std::sin(dock_yaw), shy = std::cos(dock_yaw);

    double half_wid = dock_width_ / 2.0;
    double clearance = inflation_radius_ + berth_standoff_;

    // Wall direction sign: +1 for port (+short), -1 for starboard (-short)
    double wall_sign = (dock_side_ == "port") ? 1.0 : -1.0;

    // Wall surface = centroid + wall_sign * half_wid * short_axis
    double wall_x = cx + wall_sign * half_wid * shx;
    double wall_y = cy + wall_sign * half_wid * shy;

    // Goal = wall + wall_sign * clearance * short_axis + berth_long * long_axis
    // (further from dock in the same direction as the wall)
    gx = wall_x + wall_sign * clearance * shx + berth_long_ * lx;
    gy = wall_y + wall_sign * clearance * shy + berth_long_ * ly;

    // Bow direction and staging
    if (bow_direction_ == "+x")
    {
      gyaw = dock_yaw;
      // Staging behind bow = in -long direction
      sx = gx - staging_dist_ * lx;
      sy = gy - staging_dist_ * ly;
    }
    else if (bow_direction_ == "-x")
    {
      gyaw = wrapAngle(dock_yaw + M_PI);
      // Staging in +long direction (above goal) AND offset further from
      // dock in wall_sign*short direction. This puts staging in open water
      // past the dock end. A* routes robot to staging (going past the dock),
      // then DOCKING leg A* path comes back down alongside the wall = U-turn.
      sx = gx + staging_dist_ * lx + wall_sign * (staging_dist_ * 0.5) * shx;
      sy = gy + staging_dist_ * ly + wall_sign * (staging_dist_ * 0.5) * shy;
    }
    else if (bow_direction_ == "-y")
    {
      // Bow points away from dock wall
      gyaw = wrapAngle(dock_yaw + wall_sign * (-M_PI_2));

      // Staging is further from the dock wall than the goal,
      // so the boat approaches the goal heading TOWARD the wall.
      // That way the goal is always AHEAD of the boat.
      sx = gx + wall_sign * staging_dist_ * shx;
      sy = gy + wall_sign * staging_dist_ * shy;
    }
    else
    {
      ROS_ERROR_THROTTLE(5.0, "[DockApproachPlanner] Unknown bow_direction '%s'",
                         bow_direction_.c_str());
      return;
    }

    syaw = gyaw;

    ROS_INFO_THROTTLE(5.0,
      "[DockApproachPlanner] dock=(%.1f,%.1f) yaw=%.1fdeg len=%.1f wid=%.1f side=%s bow=%s",
      cx, cy, dock_yaw*180/M_PI, dock_length_, dock_width_,
      dock_side_.c_str(), bow_direction_.c_str());
    ROS_INFO_THROTTLE(5.0,
      "[DockApproachPlanner] wall=(%.1f,%.1f) goal=(%.1f,%.1f) yaw=%.1fdeg staging=(%.1f,%.1f)",
      wall_x, wall_y, gx, gy, gyaw*180/M_PI, sx, sy);
  }

  void publishEmpty()
  {
    nav_msgs::Path e;
    e.header.frame_id = map_frame_;
    e.header.stamp = ros::Time::now();
    path_pub_.publish(e);
  }

  nav_msgs::Path planPath(double x0, double y0, double x1, double y1, double fyaw)
  {
    if (have_map_) {
      auto pts = astarSearch(map_, x0, y0, x1, y1, lethal_thresh_);
      if (!pts.empty()) {
        ROS_INFO_THROTTLE(5.0, "[DockApproachPlanner] A* path: %zu pts", pts.size());
        return worldPathToNav(map_frame_, pts, fyaw, path_downsample_);
      }
      ROS_WARN_THROTTLE(5.0, "[DockApproachPlanner] A* failed, straight line fallback");
    }
    return straightPath(map_frame_, x0, y0, x1, y1, fyaw);
  }

  void update()
  {
    if (!dock_received_) return;
    if (!dims_received_) {
      ROS_WARN_THROTTLE(5.0, "[DockApproachPlanner] Waiting for dock_dims...");
      return;
    }

    if (state_ == WAITING) {
      double elapsed = (ros::Time::now() - first_dock_time_).toSec();
      ROS_INFO_THROTTLE(2.0, "[DockApproachPlanner] WAITING %.1f/%.1f", elapsed, start_delay_);
      if (elapsed >= start_delay_) {
        state_ = APPROACHING;
        last_path_time_ = ros::Time(0);
      }
      return;
    }

    if ((ros::Time::now() - last_dock_time_).toSec() > dock_timeout_) {
      ROS_WARN_THROTTLE(3.0, "[DockApproachPlanner] Dock stale.");
      return;
    }

    double rx, ry, ryaw;
    if (!getRobotPose(rx, ry, ryaw)) return;

    double gx, gy, gyaw, sx, sy, syaw;
    computeGoalAndStaging(gx, gy, gyaw, sx, sy, syaw);

    double ds = std::hypot(rx-sx, ry-sy);
    double dg = std::hypot(rx-gx, ry-gy);
    bool pub = (ros::Time::now() - last_path_time_).toSec() >= path_interval_;

    switch (state_)
    {
      case APPROACHING:
        if (pub) {
          auto p = planPath(rx, ry, sx, sy, syaw);
          path_pub_.publish(p);
          last_path_time_ = ros::Time::now();
          ROS_INFO("[DockApproachPlanner] APPROACH path: %zu poses, dist=%.1f",
                   p.poses.size(), ds);
        }
        ROS_INFO_THROTTLE(2.0, "[DockApproachPlanner] APPROACHING dist=%.1f", ds);
        if (ds < staging_tol_) {
          state_ = DOCKING;
          last_path_time_ = ros::Time(0);
          ROS_INFO("[DockApproachPlanner] -> DOCKING");
        }
        break;

      case DOCKING:
        if (pub) {
          nav_msgs::Path p;
          if (bow_direction_ == "-y")
          {
            // Stern docking: path with ALL headings = gyaw (away from wall).
            // Guidance sees gamma_p away from wall, boat faces toward wall
            // → |chi_err|>135° → reverse mode → backs up stern-first.
            p = straightPath(map_frame_, rx, ry, gx, gy, gyaw);
            for (auto& pose : p.poses)
            {
              tf2::Quaternion q;
              q.setRPY(0.0, 0.0, gyaw);
              pose.pose.orientation = tf2::toMsg(q);
            }
          }
          else
          {
            p = planPath(rx, ry, gx, gy, gyaw);
          }
          path_pub_.publish(p);
          last_path_time_ = ros::Time::now();
          ROS_INFO("[DockApproachPlanner] DOCKING path: %zu poses, dist=%.1f",
                   p.poses.size(), dg);
        }
        ROS_INFO_THROTTLE(1.0, "[DockApproachPlanner] DOCKING dist=%.1f", dg);
        if (dg < goal_tolerance_) {
          state_ = DOCKED;
          publishEmpty();
          std_msgs::Bool m; m.data = true;
          docked_pub_.publish(m);
          ROS_INFO("[DockApproachPlanner] *** DOCKED ***");
        }
        break;

      case DOCKED:
        publishEmpty();
        break;

      default: break;
    }
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber dock_sub_, dims_sub_, map_sub_;
  ros::Publisher path_pub_, docked_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  geometry_msgs::PoseStamped dock_pose_;
  nav_msgs::OccupancyGrid map_;
  bool have_map_, dock_received_ = false, dims_received_ = false;
  ros::Time first_dock_time_, last_dock_time_, last_path_time_;
  double dock_length_ = 0, dock_width_ = 0;

  State state_;
  std::string map_frame_, dock_side_, bow_direction_;
  double start_delay_, berth_standoff_, inflation_radius_, berth_long_;
  double staging_dist_, staging_tol_, goal_tolerance_;
  double dock_timeout_, path_interval_, path_downsample_;
  int lethal_thresh_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "dock_approach_planner");
  DockApproachPlanner node;
  return 0;
}