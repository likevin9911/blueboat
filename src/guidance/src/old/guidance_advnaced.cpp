/**
 * guidance.cpp  (cursor-based, heuristics-free)
 *
 * Lookahead path-following guidance. Reads desired direction and
 * curvature directly from the planner's output (no inference from
 * heading error or yaw deltas).
 *
 * Path consumption model:
 *   The path is NOT erased as the boat travels along it. Instead, an
 *   integer cursor m_cursor tracks the closest pose found so far. The
 *   cursor only moves forward (never backward) and is searched within a
 *   small window from its previous position to avoid jumping when the
 *   path doubles back on itself (which it does for Reeds-Shepp / HC-RS
 *   paths around cusps). The cursor resets to 0 when a new path is
 *   received.
 *
 * Direction:
 *   pose.position.z carries steering::State::d ∈ {-1, 0, +1} from the
 *   steering function: forward, transition, reverse. Read directly.
 *
 * Curvature:
 *   The matching std_msgs/Float32MultiArray on ~planned_curvatures
 *   indexes 1:1 with the path. |kappa| > kappa_turn_threshold shrinks
 *   the lookahead and caps the speed.
 */

#include <guidance/guidance.h>

#include <std_msgs/Float32MultiArray.h>

#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <usv_msgs/SpeedCourse.h>

namespace blueboat_coverage
{

Guidance::Guidance()
{
  ros::NodeHandle nh;
  ros::NodeHandle nhP("~");

  m_maxSpeed           = nhP.param("max_speed", 1.5);
  m_maxSpeedTurn       = nhP.param("max_speed_turn", 0.6);
  m_minSpeed           = nhP.param("min_speed", 0.0);
  m_kappaTurnThreshold = nhP.param("kappa_turn_threshold", 0.2);
  m_goalTolerance      = nhP.param("goal_tolerance", 0.5);
  m_searchWindow       = nhP.param("search_window", 50);

  std::string path_topic, curvature_topic;
  nhP.param<std::string>("path_topic",      path_topic,      "planned_path");
  nhP.param<std::string>("curvature_topic", curvature_topic, "planned_curvatures");

  ros::Subscriber pathSub =
      nh.subscribe(path_topic, 10, &Guidance::newPath, this);
  ros::Subscriber kappaSub =
      nh.subscribe(curvature_topic, 10, &Guidance::newCurvatures, this);

  ROS_INFO_STREAM("[guidance] path topic:      " << path_topic);
  ROS_INFO_STREAM("[guidance] curvature topic: " << curvature_topic);

  m_controllerPub =
      nh.advertise<usv_msgs::SpeedCourse>("speed_heading", 10);

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  ros::Rate rate(10.0);
  while (nh.ok())
  {
    geometry_msgs::TransformStamped tfStamped;
    try
    {
      tfStamped = tfBuffer.lookupTransform("map", "base_link",
                                           ros::Time(0.0),
                                           ros::Duration(0.0));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN_THROTTLE(2.0,
          "Transform from map to base_link not found: %s", ex.what());
      ros::Duration(0.1).sleep();
      ros::spinOnce();
      continue;
    }
    double x   = tfStamped.transform.translation.x;
    double y   = tfStamped.transform.translation.y;
    double psi = tf2::getYaw(tfStamped.transform.rotation);

    followPath(x, y, psi);

    ros::spinOnce();
    rate.sleep();
  }
}

Guidance::~Guidance() {}

void Guidance::newPath(const nav_msgs::Path& path)
{
  m_path   = path;
  m_cursor = 0;
  m_arrived = false;
  ROS_INFO_STREAM("[guidance] new path received with "
                  << path.poses.size() << " poses");

  // Dump first 3 and last 3 poses so we can see what HC-RS produced.
  auto logPose = [&](size_t i)
  {
    if (i >= path.poses.size()) return;
    const auto& p = path.poses[i];
    double yaw = std::atan2(2.0 * (p.pose.orientation.w * p.pose.orientation.z +
                                   p.pose.orientation.x * p.pose.orientation.y),
                            1.0 - 2.0 * (p.pose.orientation.y * p.pose.orientation.y +
                                         p.pose.orientation.z * p.pose.orientation.z));
    ROS_INFO("[guidance] pose[%zu]: x=%.2f y=%.2f theta=%.1f deg  d=%.2f",
             i, p.pose.position.x, p.pose.position.y,
             yaw * 180.0 / M_PI, p.pose.position.z);
  };
  logPose(0);
  logPose(1);
  logPose(2);
  if (path.poses.size() > 6) {
    logPose(path.poses.size() / 2);
    logPose(path.poses.size() - 3);
    logPose(path.poses.size() - 2);
    logPose(path.poses.size() - 1);
  }
}

void Guidance::newCurvatures(const std_msgs::Float32MultiArray& msg)
{
  m_curvatures = msg.data;
}

void Guidance::followPath(double x, double y, double psi)
{
  if (m_path.poses.size() < 2)
  {
    publishStop(psi);
    return;
  }

  const auto& finalPose = m_path.poses.back();
  double dxf = finalPose.pose.position.x - x;
  double dyf = finalPose.pose.position.y - y;
  double distToGoal = std::hypot(dxf, dyf);

  // Arrival latch
  bool atFinalPose = (m_cursor >= m_path.poses.size() - 1);
  bool withinTol   = (distToGoal < m_goalTolerance);
  if (!m_arrived && (atFinalPose || withinTol))
  {
    m_arrived = true;
    ROS_INFO("[guidance] reached final waypoint (cursor=%zu, dist=%.2f)",
             m_cursor, distToGoal);
  }

  // Once arrived, station-keep toward the final pose. If wind/current/
  // overshoot pushes us past the goal, drive back toward it at a slow
  // creep. Stop completely inside a small dead-zone to avoid oscillation.
  if (m_arrived)
  {
    stationKeep(x, y, psi, finalPose);
    return;
  }

  while (m_cursor + 1 < m_path.poses.size())
  {
    const auto& a = m_path.poses[m_cursor].pose.position;
    const auto& b = m_path.poses[m_cursor + 1].pose.position;
    double sx = b.x - a.x, sy = b.y - a.y;
    double seg2 = sx * sx + sy * sy;
    if (seg2 < 1e-9) { ++m_cursor; continue; }
    double t = ((x - a.x) * sx + (y - a.y) * sy) / seg2;
    if (t > 1.0) ++m_cursor;
    else break;
  }
  // Forward-only cursor. Search closest pose only within a small forward
  // window so the cursor never jumps backward, and so it never re-locks
  // onto an earlier section if the path crosses itself (RS / HC-RS).
  size_t lo = m_cursor;
  size_t hi = std::min(m_path.poses.size(),
                       m_cursor + static_cast<size_t>(m_searchWindow));

  size_t best = m_cursor;
  double bestD2 = std::numeric_limits<double>::max();
  for (size_t i = lo; i < hi; ++i)
  {
    const auto& p = m_path.poses[i];
    double dx = x - p.pose.position.x;
    double dy = y - p.pose.position.y;
    double d2 = dx * dx + dy * dy;
    if (d2 < bestD2)
    {
      bestD2 = d2;
      best   = i;
    }
  }
  m_cursor = best;

  geometry_msgs::PoseStamped pose_d = m_path.poses[m_cursor];

  // ── Direction directly from planner output ──
  // pose.position.z holds steering::State::d ∈ {-1, 0, +1}.
  // 0 means transition / cusp; look ahead to the next nonzero value.
  bool reverseMode = false;
  {
    double d = pose_d.pose.position.z;
    if (std::abs(d) < 0.5)
    {
      for (size_t i = m_cursor + 1; i < m_path.poses.size(); ++i)
      {
        double dd = m_path.poses[i].pose.position.z;
        if (std::abs(dd) > 0.5) { d = dd; break; }
      }
    }
    reverseMode = (d < 0.0);
  }

  // ── Path tangential angle and cross-track error ──
  double gamma_p = tf2::getYaw(pose_d.pose.orientation);
  double y_e = -(x - pose_d.pose.position.x) * std::sin(gamma_p) +
               (y - pose_d.pose.position.y) * std::cos(gamma_p);

  // ── Curvature-based isTurning / lookahead shrink ──
  double kappa_here = 0.0;
  if (m_cursor < m_curvatures.size())
  {
    kappa_here = static_cast<double>(m_curvatures[m_cursor]);
  }
  bool isTurning = std::fabs(kappa_here) > m_kappaTurnThreshold;

  double delta_y_e =
      (delta_max - delta_min) * std::exp(-delta_k * std::pow(y_e, 2)) +
      delta_min;
  if (isTurning) delta_y_e = delta_min;

  // ── Course command ──
  // gamma_p is the body orientation from the steering function.
  // For HC-RS, theta is the body angle (bow direction); direction of
  // travel is encoded separately in 'd'. So in BOTH forward and reverse
  // we want the bow to point along gamma_p; sign of motion is set by
  // the sign of u below.
  double chi_r = std::atan(-y_e / delta_y_e);
  double chi_d = gamma_p + chi_r;
  // Wrap to [-π, π]. The controller's heading PID computes
  // psi_tilde = psi_slam - psi_d and wraps THAT to [-π, π], which
  // assumes both inputs are already in [-π, π]. Sending psi_d outside
  // that range produces the wrong shortest-rotation direction and the
  // boat oscillates instead of converging.
  while (chi_d >  M_PI) chi_d -= 2 * M_PI;
  while (chi_d < -M_PI) chi_d += 2 * M_PI;

  double chi_err = chi_d - psi;
  while (chi_err >  M_PI) chi_err -= 2 * M_PI;
  while (chi_err < -M_PI) chi_err += 2 * M_PI;

  // ── Speed command ──
  double u = m_maxSpeed * (1.0 - std::abs(y_e) / 5.0
                                - std::abs(chi_err) / M_PI_2);
  u = std::max(u, 0.0);

  // Slow down on approach: scale by remaining-path fraction in the last
  // few path-resolutions worth of poses. This makes the boat actually
  // arrive at the final pose instead of crashing past it.
  size_t remaining = (m_cursor < m_path.poses.size())
                         ? (m_path.poses.size() - 1 - m_cursor)
                         : 0;
  const size_t kSlowdownPoses = 40;  // ~2 m at 0.05 m path resolution
  if (remaining < kSlowdownPoses)
  {
    double frac = static_cast<double>(remaining) / kSlowdownPoses;
    u *= frac;
    u = std::max(u, 0.15);
  }
  else
  {
    // Far from the goal: enforce min_speed so the boat doesn't stall on
    // tiny path tangent errors. Near the goal the floor is removed so
    // arrival is graceful.
    u = std::max(u, m_minSpeed);
  }

  if (isTurning)   u = std::min(u, m_maxSpeedTurn);
  if (reverseMode) u = -u;

  usv_msgs::SpeedCourse msg;
  msg.speed  = u;
  msg.course = chi_d;
  m_controllerPub.publish(msg);

  ROS_INFO_STREAM_THROTTLE(0.5,
      "[guidance] cursor=" << m_cursor << "/" << m_path.poses.size()
      << " gamma_p=" << gamma_p
      << " chi_d=" << chi_d << " psi=" << psi
      << " u_d=" << u
      << " kappa=" << kappa_here
      << (reverseMode ? " [REV]" : " [FWD]"));
}

void Guidance::publishStop(double psi)
{
  usv_msgs::SpeedCourse msg;
  msg.speed  = 0.0;
  msg.course = psi;
  m_controllerPub.publish(msg);
}

void Guidance::stationKeep(double x, double y, double psi,
                           const geometry_msgs::PoseStamped& goal)
{
  double dx = goal.pose.position.x - x;
  double dy = goal.pose.position.y - y;
  double dist = std::hypot(dx, dy);

  const double deadzone     = 0.05;  // meters — position tolerance
  const double yawTolerance = 3.0 * M_PI / 180.0;  // 3 deg — final yaw tolerance

  // Inside position deadzone: rotate in place to match the goal pose's yaw.
  // The controller's surge PID with speed=0 will hold the boat near the goal
  // while the yaw PID drives heading to goalYaw.
  if (dist < deadzone)
  {
    double goalYaw = tf2::getYaw(goal.pose.orientation);
    double yawErr  = goalYaw - psi;
    while (yawErr >  M_PI) yawErr -= 2 * M_PI;
    while (yawErr < -M_PI) yawErr += 2 * M_PI;

    usv_msgs::SpeedCourse msg;
    msg.speed  = 0.0;
    msg.course = goalYaw;
    m_controllerPub.publish(msg);

    ROS_INFO_STREAM_THROTTLE(1.0,
        "[guidance] station-keep: at goal, yaw_err="
        << yawErr * 180.0 / M_PI << " deg"
        << (std::fabs(yawErr) < yawTolerance ? " [SETTLED]" : ""));
    return;
  }

  // Outer band: drive back toward the goal. Choose forward vs reverse by
  // which one requires less rotation from current heading. Speed scales
  // with distance, capped at a slow creep.
  double bearingToGoal = std::atan2(dy, dx);
  double headingErrFwd = bearingToGoal - psi;
  while (headingErrFwd >  M_PI) headingErrFwd -= 2 * M_PI;
  while (headingErrFwd < -M_PI) headingErrFwd += 2 * M_PI;

  bool useReverse = std::fabs(headingErrFwd) > M_PI_2;

  double cmdHeading = useReverse
                          ? bearingToGoal + M_PI
                          : bearingToGoal;
  while (cmdHeading >  M_PI) cmdHeading -= 2 * M_PI;
  while (cmdHeading < -M_PI) cmdHeading += 2 * M_PI;

  double creep = std::min(0.3, 0.4 * dist);
  if (useReverse) creep = -creep;

  usv_msgs::SpeedCourse msg;
  msg.speed  = creep;
  msg.course = cmdHeading;
  m_controllerPub.publish(msg);

  ROS_INFO_STREAM_THROTTLE(1.0,
      "[guidance] station-keep: dist=" << dist
      << " creep=" << creep
      << (useReverse ? " [REV]" : " [FWD]"));
}

} // namespace blueboat_coverage