#include <buoy_course/course_planner.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>

#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>

namespace buoy_course
{

// ============================================================
// Circuit Planner v2 — progressive circuit building
//
// Key change from v1: does NOT wait for all buoys before acting.
// Builds a circuit with whatever buoys are confirmed (min 2),
// starts following it, and rebuilds whenever a new buoy appears.
// The circuit grows as the boat discovers more buoys.
//
// Racing concepts (IAC / F1TENTH):
// 1. Global raceline computed from known track geometry
// 2. Minimum curvature approximation via arc interpolation
// 3. Dense waypoint path for cross-track error following
// 4. A* only for initial approach, not for the circuit itself
// ============================================================

CoursePlanner::CoursePlanner()
  : m_nhP("~")
  , m_tfListener(m_tfBuffer)
  , m_state(CourseState::WAITING_FOR_BUOYS)
  , m_currentMarkIdx(0)
  , m_robotX(0.0), m_robotY(0.0), m_robotPsi(0.0)
  , m_sequenceLocked(false)
  , m_courseCenterX(0.0), m_courseCenterY(0.0)
  , m_currentWaypointIdx(0)
  , m_lapStartIdx(0)
  , m_lapCompleted(false)
  , m_currentLap(0)
  , m_lastBuoyCount(0)
{
  m_mapFrame       = m_nhP.param<std::string>("map_frame",  "map");
  m_goalTolerance  = m_nhP.param("goal_tolerance",          3.0);
  m_roundingOffset = m_nhP.param("rounding_offset",         5.0);
  m_startX         = m_nhP.param("start_x",                 0.0);
  m_startY         = m_nhP.param("start_y",                 0.0);
  m_buoyCount      = m_nhP.param("buoy_count",              4);
  m_planRate       = m_nhP.param("plan_rate",               2.0);

  // Circuit parameters
  m_totalLaps      = m_nhP.param("total_laps",              1);
  m_currentLap     = 0;

  std::string rounding_side = m_nhP.param<std::string>("rounding_side", "starboard");
  m_roundCW = (rounding_side == "starboard");

  m_arcPoints = m_nhP.param("arc_points_per_buoy", 8);

  // Minimum buoys needed to start building a circuit
  m_minBuoysForCircuit = m_nhP.param("min_buoys_for_circuit", 2);

  m_buoySub = m_nh.subscribe("buoy_map",    10, &CoursePlanner::buoyCb, this);
  m_mapSub  = m_nh.subscribe("inflated_map", 1, &CoursePlanner::mapCb,  this);

  m_pathPub     = m_nh.advertise<nav_msgs::Path>("simple_dubins_path", 10);
  m_activeIdPub = m_nh.advertise<std_msgs::Int32>("active_buoy_id",    10);
  m_statusPub   = m_nh.advertise<std_msgs::Bool>("lap_finished",       10, true);

  m_planTimer = m_nh.createTimer(
      ros::Duration(1.0 / m_planRate), &CoursePlanner::timerCb, this);

  ROS_INFO("[CircuitPlanner] Started. buoys=%d  min_start=%d  offset=%.2f  "
           "laps=%d  rounding=%s  arc_pts=%d",
           m_buoyCount, m_minBuoysForCircuit, m_roundingOffset,
           m_totalLaps, rounding_side.c_str(), m_arcPoints);
}

// ============================================================
// buoyCb — progressive: always accept buoy updates
// ============================================================
void CoursePlanner::buoyCb(const geometry_msgs::PoseArray::ConstPtr& msg)
{
  if (msg->poses.empty()) return;

  int old_count = static_cast<int>(m_confirmedBuoys.size());

  if (m_sequenceLocked)
  {
    // Post-lock: update positions in-place, don't reorder
    for (int i = 0; i < static_cast<int>(m_confirmedBuoys.size()); i++)
    {
      double best_d = 4.0;
      for (const auto& inc : msg->poses)
      {
        double d = std::hypot(
            inc.position.x - m_confirmedBuoys[i].position.x,
            inc.position.y - m_confirmedBuoys[i].position.y);
        if (d < best_d)
        {
          best_d = d;
          m_confirmedBuoys[i].position.x = inc.position.x;
          m_confirmedBuoys[i].position.y = inc.position.y;
        }
      }
    }
    // Rebuild circuit with updated positions periodically
    return;
  }

  // Pre-lock: accept the full buoy list from tracker
  m_confirmedBuoys = msg->poses;
  m_visited.assign(m_confirmedBuoys.size(), false);

  int new_count = static_cast<int>(m_confirmedBuoys.size());

  // If new buoys appeared, rebuild circuit
  if (new_count != old_count && new_count >= m_minBuoysForCircuit)
  {
    ROS_INFO("[CP] Buoy count changed %d -> %d, rebuilding circuit.",
             old_count, new_count);
    sortBuoys();
    buildCircuit();
    m_circuitDirty = true;
  }

  // Lock once we have all expected buoys
  if (new_count >= m_buoyCount && !m_sequenceLocked)
  {
    m_sequenceLocked = true;
    sortBuoys();
    buildCircuit();
    m_circuitDirty = true;
    ROS_INFO("[CP] *** Sequence LOCKED with %d buoys ***", new_count);
  }
}

// ============================================================
// sortBuoys — compute centroid, sort CW or CCW
// ============================================================
void CoursePlanner::sortBuoys()
{
  int n = static_cast<int>(m_confirmedBuoys.size());
  if (n < 2) return;

  // Compute centroid
  double cx = 0, cy = 0;
  for (auto& b : m_confirmedBuoys)
  { cx += b.position.x; cy += b.position.y; }
  cx /= n;
  cy /= n;
  m_courseCenterX = cx;
  m_courseCenterY = cy;

  // Sort by angle around centroid
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(),
    [&](int a, int b)
    {
      double angle_a = std::atan2(
          m_confirmedBuoys[a].position.y - cy,
          m_confirmedBuoys[a].position.x - cx);
      double angle_b = std::atan2(
          m_confirmedBuoys[b].position.y - cy,
          m_confirmedBuoys[b].position.x - cx);
      return m_roundCW ? (angle_a > angle_b) : (angle_a < angle_b);
    });

  std::vector<geometry_msgs::Pose> sorted;
  for (int i : indices)
    sorted.push_back(m_confirmedBuoys[i]);
  m_confirmedBuoys = sorted;

  ROS_INFO("[CP] Sorted %d buoys (%s). Center=(%.2f,%.2f)",
           n, m_roundCW ? "CW" : "CCW", cx, cy);
  for (int i = 0; i < n; i++)
    ROS_INFO("  Buoy[%d] (%.2f, %.2f)", i,
             m_confirmedBuoys[i].position.x,
             m_confirmedBuoys[i].position.y);
}

// ============================================================
// buildCircuit — generate the closed-loop raceline
// Works with 2+ buoys. With 2 buoys it makes an elongated oval.
// With 3+ it wraps around the convex hull.
// ============================================================
void CoursePlanner::buildCircuit()
{
  m_circuitWaypoints.clear();

  int n = static_cast<int>(m_confirmedBuoys.size());
  if (n < 2) return;

  for (int i = 0; i < n; i++)
  {
    const auto& buoy = m_confirmedBuoys[i];
    const auto& prev = m_confirmedBuoys[(i - 1 + n) % n];
    const auto& next = m_confirmedBuoys[(i + 1) % n];

    double bx = buoy.position.x;
    double by = buoy.position.y;

    // Outward direction (away from centroid)
    double inx = m_courseCenterX - bx;
    double iny = m_courseCenterY - by;
    double ind = std::hypot(inx, iny);
    if (ind < 1e-3) { inx = 1.0; iny = 0.0; ind = 1.0; }
    double outx = -inx / ind;
    double outy = -iny / ind;

    // Direction from buoy toward prev and next
    double to_prev_x = prev.position.x - bx;
    double to_prev_y = prev.position.y - by;
    double to_prev_d = std::hypot(to_prev_x, to_prev_y);
    if (to_prev_d < 1e-3) { to_prev_x = 1.0; to_prev_y = 0.0; to_prev_d = 1.0; }
    to_prev_x /= to_prev_d;
    to_prev_y /= to_prev_d;

    double to_next_x = next.position.x - bx;
    double to_next_y = next.position.y - by;
    double to_next_d = std::hypot(to_next_x, to_next_y);
    if (to_next_d < 1e-3) { to_next_x = 1.0; to_next_y = 0.0; to_next_d = 1.0; }
    to_next_x /= to_next_d;
    to_next_y /= to_next_d;

    // Arc start/end angles: blend neighbor direction with outward
    double start_angle = std::atan2(
        to_prev_y + outy, to_prev_x + outx);
    double end_angle = std::atan2(
        to_next_y + outy, to_next_x + outx);

    // Ensure arc sweeps correct direction
    double sweep = end_angle - start_angle;
    if (m_roundCW)
    {
      while (sweep > 0) sweep -= 2.0 * M_PI;
      while (sweep < -2.0 * M_PI) sweep += 2.0 * M_PI;
    }
    else
    {
      while (sweep < 0) sweep += 2.0 * M_PI;
      while (sweep > 2.0 * M_PI) sweep -= 2.0 * M_PI;
    }

    if (std::fabs(sweep) < 0.1)
    {
      geometry_msgs::PoseStamped ps;
      ps.header.frame_id = m_mapFrame;
      ps.pose.position.x = bx + outx * m_roundingOffset;
      ps.pose.position.y = by + outy * m_roundingOffset;
      ps.pose.position.z = 0.0;
      ps.pose.orientation.w = 1.0;
      m_circuitWaypoints.push_back(ps);
      continue;
    }

    for (int j = 0; j <= m_arcPoints; j++)
    {
      double t = static_cast<double>(j) / m_arcPoints;
      double angle = start_angle + sweep * t;
      double wx = bx + std::cos(angle) * m_roundingOffset;
      double wy = by + std::sin(angle) * m_roundingOffset;

      geometry_msgs::PoseStamped ps;
      ps.header.frame_id = m_mapFrame;
      ps.header.stamp    = ros::Time::now();
      ps.pose.position.x = wx;
      ps.pose.position.y = wy;
      ps.pose.position.z = 0.0;

      double yaw = m_roundCW ? (angle - M_PI_2) : (angle + M_PI_2);
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);
      ps.pose.orientation = tf2::toMsg(q);

      m_circuitWaypoints.push_back(ps);
    }
  }

  densifyCircuit(0.5);

  ROS_INFO("[CP] Circuit built: %zu waypoints, %.1fm length, %d buoys",
           m_circuitWaypoints.size(), estimateCircuitLength(), n);
}

// ============================================================
// densifyCircuit — ensure dense waypoint spacing for guidance
// ============================================================
void CoursePlanner::densifyCircuit(double target_spacing)
{
  if (m_circuitWaypoints.size() < 2) return;

  std::vector<geometry_msgs::PoseStamped> dense;
  int n = static_cast<int>(m_circuitWaypoints.size());

  for (int i = 0; i < n; i++)
  {
    const auto& p0 = m_circuitWaypoints[i];
    const auto& p1 = m_circuitWaypoints[(i + 1) % n];

    dense.push_back(p0);

    double dx = p1.pose.position.x - p0.pose.position.x;
    double dy = p1.pose.position.y - p0.pose.position.y;
    double d = std::hypot(dx, dy);

    if (d > target_spacing)
    {
      int subdivisions = static_cast<int>(std::ceil(d / target_spacing));
      for (int j = 1; j < subdivisions; j++)
      {
        double t = static_cast<double>(j) / subdivisions;

        geometry_msgs::PoseStamped ps;
        ps.header = p0.header;
        ps.pose.position.x = p0.pose.position.x + dx * t;
        ps.pose.position.y = p0.pose.position.y + dy * t;
        ps.pose.position.z = 0.0;

        double yaw = std::atan2(dy, dx);
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        ps.pose.orientation = tf2::toMsg(q);

        dense.push_back(ps);
      }
    }
  }

  m_circuitWaypoints = dense;
}

double CoursePlanner::estimateCircuitLength() const
{
  double len = 0;
  int n = static_cast<int>(m_circuitWaypoints.size());
  for (int i = 0; i < n; i++)
  {
    const auto& p0 = m_circuitWaypoints[i];
    const auto& p1 = m_circuitWaypoints[(i + 1) % n];
    len += std::hypot(p1.pose.position.x - p0.pose.position.x,
                      p1.pose.position.y - p0.pose.position.y);
  }
  return len;
}

void CoursePlanner::mapCb(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
  m_map = msg;
}

void CoursePlanner::timerCb(const ros::TimerEvent&)
{
  geometry_msgs::TransformStamped tf;
  try
  {
    tf = m_tfBuffer.lookupTransform(
        m_mapFrame, "base_link", ros::Time(0), ros::Duration(0.1));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN_THROTTLE(5.0, "[CP] TF: %s", ex.what());
    return;
  }
  m_robotX   = tf.transform.translation.x;
  m_robotY   = tf.transform.translation.y;
  m_robotPsi = tf2::getYaw(tf.transform.rotation);

  ROS_INFO_THROTTLE(3.0,
      "[CP] %s  buoys=%zu/%d  lap=%d/%d  robot=(%.1f,%.1f)  "
      "wpIdx=%d/%zu  locked=%s",
      stateName(m_state).c_str(),
      m_confirmedBuoys.size(), m_buoyCount,
      m_currentLap, m_totalLaps,
      m_robotX, m_robotY,
      m_currentWaypointIdx,
      m_circuitWaypoints.size(),
      m_sequenceLocked ? "YES" : "no");

  updateFSM(m_robotX, m_robotY, m_robotPsi);
}

// ============================================================
// FSM
// ============================================================
void CoursePlanner::updateFSM(double rx, double ry, double psi)
{
  switch (m_state)
  {
    case CourseState::WAITING_FOR_BUOYS:
    {
      if (!m_circuitWaypoints.empty())
        transitionTo(CourseState::PLANNING);
      break;
    }

    case CourseState::PLANNING:
    {
      if (m_circuitWaypoints.empty())
      {
        ROS_WARN_THROTTLE(3.0, "[CP] No circuit waypoints.");
        break;
      }

      int entry = findNearestWaypoint(rx, ry);
      m_currentWaypointIdx = entry;

      const auto& entry_wp = m_circuitWaypoints[entry];
      double dist_to_entry = std::hypot(
          rx - entry_wp.pose.position.x,
          ry - entry_wp.pose.position.y);

      // If already close, skip A* approach
      if (dist_to_entry < m_goalTolerance * 2.0)
      {
        ROS_INFO("[CP] Already near circuit (%.2fm). Starting.", dist_to_entry);
        if (m_currentLap == 0) m_currentLap = 1;
        m_lapStartIdx = m_currentWaypointIdx;
        publishCircuitPath();
        transitionTo(CourseState::ROUNDING);
        break;
      }

      // Try A* approach, fall back to direct path
      if (m_map)
      {
        nav_msgs::Path approach_path;
        if (planPath(rx, ry, entry_wp.pose.position.x,
                     entry_wp.pose.position.y, approach_path))
        {
          m_pathPub.publish(approach_path);
          ROS_INFO("[CP] A* approach to wp %d dist=%.2f",
                   entry, dist_to_entry);
        }
        else
        {
          ROS_WARN("[CP] A* failed, using direct path.");
          nav_msgs::Path direct;
          direct.header.stamp = ros::Time::now();
          direct.header.frame_id = m_mapFrame;

          geometry_msgs::PoseStamped start_ps;
          start_ps.header = direct.header;
          start_ps.pose.position.x = rx;
          start_ps.pose.position.y = ry;
          double yaw = std::atan2(entry_wp.pose.position.y - ry,
                                  entry_wp.pose.position.x - rx);
          tf2::Quaternion q; q.setRPY(0, 0, yaw);
          start_ps.pose.orientation = tf2::toMsg(q);
          direct.poses.push_back(start_ps);
          direct.poses.push_back(entry_wp);
          m_pathPub.publish(direct);
        }
        transitionTo(CourseState::NAVIGATING);
      }
      else
      {
        ROS_WARN_THROTTLE(3.0, "[CP] Waiting for map...");
      }
      break;
    }

    case CourseState::NAVIGATING:
    {
      // Circuit rebuilt? Replan approach.
      if (m_circuitDirty)
      {
        m_circuitDirty = false;
        ROS_INFO("[CP] Circuit rebuilt during approach, replanning.");
        transitionTo(CourseState::PLANNING);
        break;
      }

      const auto& target = m_circuitWaypoints[m_currentWaypointIdx];
      double dist = std::hypot(rx - target.pose.position.x,
                               ry - target.pose.position.y);

      ROS_INFO_THROTTLE(2.0, "[CP] APPROACH dist=%.2f", dist);

      if (dist < m_goalTolerance)
      {
        ROS_INFO("[CP] Reached circuit. Starting lap 1.");
        m_currentLap = 1;
        m_lapStartIdx = m_currentWaypointIdx;
        transitionTo(CourseState::ROUNDING);
        publishCircuitPath();
      }
      break;
    }

    case CourseState::ROUNDING:
    {
      if (m_circuitWaypoints.empty()) break;

      // New buoy found while circuiting? Replan with bigger circuit.
      if (m_circuitDirty && !m_sequenceLocked)
      {
        m_circuitDirty = false;
        ROS_INFO("[CP] New buoy discovered, expanding circuit.");
        transitionTo(CourseState::PLANNING);
        break;
      }
      m_circuitDirty = false;

      advanceWaypointIndex(rx, ry);

      // Continuously republish so guidance always has waypoints ahead
      static ros::Time last_circuit_pub = ros::Time(0);
        if ((ros::Time::now() - last_circuit_pub).toSec() > 5.0)
        {
          publishCircuitPath();
          last_circuit_pub = ros::Time::now();
        }

      if (m_lapCompleted)
      {
        m_lapCompleted = false;
        ROS_INFO("[CP] *** Lap %d COMPLETE ***", m_currentLap);

        std_msgs::Bool lap_msg;
        lap_msg.data = true;
        m_statusPub.publish(lap_msg);

        if (m_totalLaps > 0 && m_currentLap >= m_totalLaps)
        {
          transitionTo(CourseState::RETURNING);
        }
        else
        {
          m_currentLap++;
          ROS_INFO("[CP] Starting lap %d", m_currentLap);
        }
      }
      break;
    }

    case CourseState::RETURNING:
    {
      static bool sent = false;
      if (!sent)
      {
        nav_msgs::Path path;
        if (planPath(rx, ry, m_startX, m_startY, path))
        {
          m_pathPub.publish(path);
          ROS_INFO("[CP] Returning to start.");
          sent = true;
        }
        else
        {
          nav_msgs::Path direct;
          direct.header.stamp = ros::Time::now();
          direct.header.frame_id = m_mapFrame;
          geometry_msgs::PoseStamped ps;
          ps.header = direct.header;
          ps.pose.position.x = m_startX;
          ps.pose.position.y = m_startY;
          double yaw = std::atan2(m_startY - ry, m_startX - rx);
          tf2::Quaternion q; q.setRPY(0, 0, yaw);
          ps.pose.orientation = tf2::toMsg(q);
          direct.poses.push_back(ps);
          m_pathPub.publish(direct);
          sent = true;
        }
      }
      if (std::hypot(rx - m_startX, ry - m_startY) < m_goalTolerance)
      {
        transitionTo(CourseState::FINISHED);
        sent = false;
      }
      break;
    }

    case CourseState::FINISHED:
    {
      nav_msgs::Path empty;
      empty.header.stamp    = ros::Time::now();
      empty.header.frame_id = m_mapFrame;
      m_pathPub.publish(empty);
      std_msgs::Bool done; done.data = true;
      m_statusPub.publish(done);
      ROS_INFO_THROTTLE(10.0, "[CP] Course COMPLETE!");
      break;
    }
  }
}

void CoursePlanner::publishCircuitPath()
{
  nav_msgs::Path path;
  path.header.stamp    = ros::Time::now();
  path.header.frame_id = m_mapFrame;

  int n = static_cast<int>(m_circuitWaypoints.size());
  if (n == 0) return;

  // One full lap ahead + overlap
  int count = n + 20;
  for (int i = 0; i < count; i++)
  {
    int idx = (m_currentWaypointIdx + i) % n;
    geometry_msgs::PoseStamped ps = m_circuitWaypoints[idx];
    ps.header = path.header;
    path.poses.push_back(ps);
  }

  m_pathPub.publish(path);
}

// In advanceWaypointIndex, track cumulative distance instead:
void CoursePlanner::advanceWaypointIndex(double rx, double ry)
{
  int n = static_cast<int>(m_circuitWaypoints.size());
  if (n == 0) return;

  int search_window = std::min(n / 2, 50);
  double best_dist = std::numeric_limits<double>::max();
  int best_idx = m_currentWaypointIdx;

  for (int i = 0; i < search_window; i++)
  {
    int idx = (m_currentWaypointIdx + i) % n;
    double d = std::hypot(rx - m_circuitWaypoints[idx].pose.position.x,
                          ry - m_circuitWaypoints[idx].pose.position.y);
    if (d < best_dist)
    {
      best_dist = d;
      best_idx = idx;
    }
  }

  // Count waypoints advanced this tick
  int prev = m_currentWaypointIdx;
  m_currentWaypointIdx = best_idx;

  // Count how many waypoints we moved forward (handling wrap)
  int advanced = best_idx - prev;
  if (advanced < 0) advanced += n;  // wrapped around
  m_waypointsThisLap += advanced;

  // A lap is complete when we've traversed all waypoints
  if (m_waypointsThisLap >= n)
  {
    m_lapCompleted = true;
    m_waypointsThisLap = 0;  // reset for next lap
  }
}

int CoursePlanner::findNearestWaypoint(double rx, double ry) const
{
  int best = 0;
  double best_d = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(m_circuitWaypoints.size()); i++)
  {
    double d = std::hypot(rx - m_circuitWaypoints[i].pose.position.x,
                          ry - m_circuitWaypoints[i].pose.position.y);
    if (d < best_d) { best_d = d; best = i; }
  }
  return best;
}

void CoursePlanner::transitionTo(CourseState s)
{
  if (s == m_state) return;
  ROS_INFO("[CP] %s -> %s", stateName(m_state).c_str(), stateName(s).c_str());
  m_state = s;
}

std::string CoursePlanner::stateName(CourseState s) const
{
  switch (s)
  {
    case CourseState::WAITING_FOR_BUOYS: return "WAIT";
    case CourseState::PLANNING:          return "PLAN";
    case CourseState::NAVIGATING:        return "APPROACH";
    case CourseState::ROUNDING:          return "CIRCUIT";
    case CourseState::RETURNING:         return "RETURN";
    case CourseState::FINISHED:          return "DONE";
    default:                            return "?";
  }
}

void CoursePlanner::publishActiveBuoyId(int id)
{
  std_msgs::Int32 msg; msg.data = id;
  m_activeIdPub.publish(msg);
}

int CoursePlanner::nextUnvisitedIdx() const
{
  for (int i = 0; i < static_cast<int>(m_confirmedBuoys.size()); i++)
    if (i >= static_cast<int>(m_visited.size()) || !m_visited[i])
      return i;
  return -1;
}

int CoursePlanner::visitedCount() const
{
  return static_cast<int>(std::count(m_visited.begin(), m_visited.end(), true));
}

// ============================================================
// A*
// ============================================================
bool CoursePlanner::planPath(double sx, double sy,
                              double gx_w, double gy_w,
                              nav_msgs::Path& path_out)
{
  if (!m_map) { ROS_ERROR("[CP] No map."); return false; }

  int sx_g, sy_g, gx_g, gy_g;
  if (!worldToGrid(sx, sy, sx_g, sy_g) ||
      !worldToGrid(gx_w, gy_w, gx_g, gy_g))
  {
    ROS_WARN("[CP] Start or goal outside map.");
    return false;
  }

  if (!isFree(gx_g, gy_g))
  {
    bool found = false;
    for (int r = 1; r < 50 && !found; r++)
      for (int ddx = -r; ddx <= r && !found; ddx++)
        for (int ddy = -r; ddy <= r && !found; ddy++)
          if ((std::abs(ddx)==r || std::abs(ddy)==r) &&
              isFree(gx_g+ddx, gy_g+ddy))
          { gx_g+=ddx; gy_g+=ddy; found=true; }
    if (!found) { ROS_ERROR("[CP] Goal occupied."); return false; }
  }

  int W = m_map->info.width, H = m_map->info.height;
  auto idx = [W](int x, int y){ return y*W+x; };

  std::vector<float> g(W*H, std::numeric_limits<float>::max());
  std::vector<bool>  cl(W*H, false);
  std::vector<std::pair<int,int>> par(W*H, {-1,-1});

  using PQ = std::priority_queue<AStarNode,std::vector<AStarNode>,
                                  std::greater<AStarNode>>;
  PQ open;
  g[idx(sx_g,sy_g)]=0;
  AStarNode s0; s0.x=sx_g; s0.y=sy_g; s0.g=0;
  s0.h=heuristic(sx_g,sy_g,gx_g,gy_g); open.push(s0);

  const int   DX[]={1,-1,0,0,1,1,-1,-1};
  const int   DY[]={0,0,1,-1,1,-1,1,-1};
  const float DC[]={1,1,1,1,1.414f,1.414f,1.414f,1.414f};

  bool found=false; int iters=0;
  while (!open.empty() && iters++<W*H)
  {
    AStarNode cur=open.top(); open.pop();
    if (cl[idx(cur.x,cur.y)]) continue;
    cl[idx(cur.x,cur.y)]=true;
    if (cur.x==gx_g&&cur.y==gy_g){found=true;break;}
    for (int i=0;i<8;i++)
    {
      int nx=cur.x+DX[i], ny=cur.y+DY[i];
      if (nx<0||nx>=W||ny<0||ny>=H) continue;
      if (!isFree(nx,ny)||cl[idx(nx,ny)]) continue;
      float ng=g[idx(cur.x,cur.y)]+DC[i];
      if (ng<g[idx(nx,ny)])
      {
        g[idx(nx,ny)]=ng; par[idx(nx,ny)]={cur.x,cur.y};
        AStarNode nb; nb.x=nx; nb.y=ny; nb.g=ng;
        nb.h=heuristic(nx,ny,gx_g,gy_g); open.push(nb);
      }
    }
  }
  if (!found){ROS_WARN("[CP] A* no path.");return false;}

  std::vector<std::pair<int,int>> cells;
  int cx=gx_g, cy=gy_g;
  while (cx!=-1){cells.push_back({cx,cy}); auto p=par[idx(cx,cy)]; cx=p.first; cy=p.second;}
  std::reverse(cells.begin(),cells.end());

  path_out.header.stamp=ros::Time::now();
  path_out.header.frame_id=m_mapFrame;
  path_out.poses.clear();
  for (size_t i=0;i<cells.size();i++)
  {
    double wx,wy; gridToWorld(cells[i].first,cells[i].second,wx,wy);
    double yaw=0;
    if (i+1<cells.size()){double nx2,ny2; gridToWorld(cells[i+1].first,cells[i+1].second,nx2,ny2); yaw=std::atan2(ny2-wy,nx2-wx);}
    else if (i>0){double px2,py2; gridToWorld(cells[i-1].first,cells[i-1].second,px2,py2); yaw=std::atan2(wy-py2,wx-px2);}
    geometry_msgs::PoseStamped ps; ps.header=path_out.header;
    ps.pose.position.x=wx; ps.pose.position.y=wy; ps.pose.position.z=0;
    tf2::Quaternion q; q.setRPY(0,0,yaw); ps.pose.orientation=tf2::toMsg(q);
    path_out.poses.push_back(ps);
  }
  return true;
}

bool CoursePlanner::worldToGrid(double wx, double wy, int& gx, int& gy) const
{
  if (!m_map) return false;
  double res=m_map->info.resolution;
  gx=static_cast<int>((wx-m_map->info.origin.position.x)/res);
  gy=static_cast<int>((wy-m_map->info.origin.position.y)/res);
  return gx>=0&&gx<static_cast<int>(m_map->info.width)&&
         gy>=0&&gy<static_cast<int>(m_map->info.height);
}

bool CoursePlanner::gridToWorld(int gx, int gy, double& wx, double& wy) const
{
  if (!m_map) return false;
  wx=m_map->info.origin.position.x+(gx+0.5)*m_map->info.resolution;
  wy=m_map->info.origin.position.y+(gy+0.5)*m_map->info.resolution;
  return true;
}

bool CoursePlanner::isFree(int gx, int gy) const
{
  if (!m_map) return false;
  if (gx<0||gx>=static_cast<int>(m_map->info.width)||
      gy<0||gy>=static_cast<int>(m_map->info.height)) return false;
  return m_map->data[gy*m_map->info.width+gx] == 0;
}

float CoursePlanner::heuristic(int ax, int ay, int bx, int by) const
{
  float dx=static_cast<float>(bx-ax), dy=static_cast<float>(by-ay);
  return std::sqrt(dx*dx+dy*dy);
}

} // namespace buoy_course

int main(int argc, char** argv)
{
  ros::init(argc, argv, "course_planner_node");
  buoy_course::CoursePlanner node;
  ros::spin();
  return 0;
}