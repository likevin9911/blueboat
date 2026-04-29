/**
 * regulated_pure_pursuit_guidance.cpp
 *
 * USV port of nav2_regulated_pure_pursuit_controller. The
 * use_dynamic_window flag selects between two ways of converting the
 * regulated linear velocity and the pure-pursuit curvature into the
 * final (v, w):
 *
 *   use_dynamic_window=false  (default):
 *     w = v * kappa, then publish.
 *
 *   use_dynamic_window=true:
 *     pass (v_regulated, kappa, current_speed, accel limits) into
 *     dynamic_window_pure_pursuit::computeDynamicWindowVelocities, which
 *     enforces acceleration feasibility one timestep ahead. This is the
 *     mode that prevents overshoot on tight turns: when the upcoming
 *     curvature spikes, the regulator collapses v_regulated, the dynamic
 *     window caps how fast we can decelerate, and the boat starts braking
 *     before the curve rather than at the apex.
 *
 * (v, w) is then converted to (speed, course) by projecting heading
 * forward over a short preview horizon, since the existing controller
 * tracks course not yaw rate.
 */

#include <guidance/regulated_pure_pursuit_guidance.h>
#include <dynamic_window_pure_pursuit_functions/dynamic_window_pure_pursuit_functions.h>

#include <usv_msgs/SpeedCourse.h>

#include <tf2/utils.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace blueboat_coverage
{

RegulatedPurePursuitGuidance::RegulatedPurePursuitGuidance()
{
  ros::NodeHandle nh;
  ros::NodeHandle nhP("~");

  // ── Mode ──
  m_useDynamicWindow = nhP.param("use_dynamic_window", false);

  // ── Lookahead ──
  m_useVelocityScaledLookahead =
      nhP.param("use_velocity_scaled_lookahead_dist", true);
  m_lookaheadDist     = nhP.param("lookahead_dist",     1.0);
  m_minLookaheadDist  = nhP.param("min_lookahead_dist", 0.5);
  m_maxLookaheadDist  = nhP.param("max_lookahead_dist", 3.0);
  m_lookaheadTime     = nhP.param("lookahead_time",     1.5);

  // ── Curvature regulation ──
  m_useRegulatedLinearVelocityScaling =
      nhP.param("use_regulated_linear_velocity_scaling", true);
  m_regulatedLinearScalingMinRadius =
      nhP.param("regulated_linear_scaling_min_radius", 1.5);
  m_regulatedLinearScalingMinSpeed =
      nhP.param("regulated_linear_scaling_min_speed", 0.25);

  // ── Heading regulation ──
  m_useHeadingRegulation = nhP.param("use_heading_regulation", true);
  m_headingScalingThreshold =
      nhP.param("heading_scaling_threshold", 0.5);  // ~28 deg

  // ── Approach velocity scaling ──
  m_approachVelocityScalingDist =
      nhP.param("approach_velocity_scaling_dist", 2.0);
  m_minApproachLinearVelocity =
      nhP.param("min_approach_linear_velocity", 0.1);

  // ── Speed limits ──
  m_maxLinearVel  = nhP.param("max_linear_vel",   1.5);
  m_minLinearVel  = nhP.param("min_linear_vel",  -0.6);
  m_maxAngularVel = nhP.param("max_angular_vel",  1.0);
  m_minAngularVel = nhP.param("min_angular_vel", -1.0);

  // ── Acceleration limits (define the dynamic window) ──
  m_maxLinearAccel  = nhP.param("max_linear_accel",   0.5);
  m_maxLinearDecel  = nhP.param("max_linear_decel",  -1.0);
  m_maxAngularAccel = nhP.param("max_angular_accel",  1.5);
  m_maxAngularDecel = nhP.param("max_angular_decel", -2.5);

  // ── Arrival / cursor ──
  m_goalTolerance = nhP.param("goal_tolerance", 0.5);
  m_searchWindow  = nhP.param("search_window",  50);

  m_controlFrequency = nhP.param("control_frequency", 10.0);

  // ── Topics ──
  std::string path_topic, curvature_topic, odom_topic;
  nhP.param<std::string>("path_topic",      path_topic,      "planned_path");
  nhP.param<std::string>("curvature_topic", curvature_topic, "planned_curvatures");
  nhP.param<std::string>("odom_topic",      odom_topic,      "/odometry/filtered");

  m_pathSub  = nh.subscribe(path_topic,      10,
                            &RegulatedPurePursuitGuidance::newPath, this);
  m_kappaSub = nh.subscribe(curvature_topic, 10,
                            &RegulatedPurePursuitGuidance::newCurvatures, this);

  // Only subscribe to odometry when dynamic window mode is on. When off,
  // the odom feedback isn't used for anything and the subscription is
  // pointless overhead.
  if (m_useDynamicWindow)
  {
    m_odomSub = nh.subscribe(odom_topic, 10,
                             &RegulatedPurePursuitGuidance::newOdom, this);
  }

  m_controllerPub =
      nh.advertise<usv_msgs::SpeedCourse>("speed_heading", 10);
  m_carrotPub =
      nhP.advertise<geometry_msgs::PoseStamped>("carrot", 10);

  ROS_INFO_STREAM("[rpp_guidance] mode: "
                  << (m_useDynamicWindow ? "RPP+DWPP" : "RPP"));
  ROS_INFO_STREAM("[rpp_guidance] path topic:      " << path_topic);
  ROS_INFO_STREAM("[rpp_guidance] curvature topic: " << curvature_topic);
  if (m_useDynamicWindow)
  {
    ROS_INFO_STREAM("[rpp_guidance] odom topic:      " << odom_topic);
    ROS_INFO("[rpp_guidance] v in [%.2f, %.2f] m/s, w in [%.2f, %.2f] rad/s",
             m_minLinearVel, m_maxLinearVel,
             m_minAngularVel, m_maxAngularVel);
    ROS_INFO("[rpp_guidance] accel: lin a=%.2f d=%.2f, ang a=%.2f d=%.2f",
             m_maxLinearAccel, m_maxLinearDecel,
             m_maxAngularAccel, m_maxAngularDecel);
  }

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);

  ros::Rate rate(m_controlFrequency);
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
          "[rpp_guidance] map->base_link tf not found: %s", ex.what());
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

RegulatedPurePursuitGuidance::~RegulatedPurePursuitGuidance() {}

// ──────────────────────────────────────────────────────────────────────
// Subscribers
// ──────────────────────────────────────────────────────────────────────

void RegulatedPurePursuitGuidance::newPath(const nav_msgs::Path& path)
{
  m_path           = path;
  m_cursor         = 0;
  m_arrived        = false;
  // m_stopPublished  = false;
  ROS_INFO_STREAM("[rpp_guidance] new path: " << path.poses.size() << " poses");
}

void RegulatedPurePursuitGuidance::newCurvatures(
    const std_msgs::Float32MultiArray& msg)
{
  m_curvatures = msg.data;
}

void RegulatedPurePursuitGuidance::newOdom(const nav_msgs::Odometry& msg)
{
  m_currentVel.linear.x  = msg.twist.twist.linear.x;
  m_currentVel.angular.z = msg.twist.twist.angular.z;
  m_haveOdom = true;
}

// ──────────────────────────────────────────────────────────────────────
// Top-level step
// ──────────────────────────────────────────────────────────────────────

void RegulatedPurePursuitGuidance::followPath(double x, double y, double psi)
{
  if (m_path.poses.size() < 2)
  {
    publishStop(psi);
    return;
  }

  // ── Update cursor (forward-only, windowed search) ──
  std::size_t lo = m_cursor;
  std::size_t hi = std::min(m_path.poses.size(),
                            m_cursor + static_cast<std::size_t>(m_searchWindow));
  std::size_t best = m_cursor;
  double bestD2 = std::numeric_limits<double>::max();
  for (std::size_t i = lo; i < hi; ++i)
  {
    const auto& p = m_path.poses[i];
    double dx = x - p.pose.position.x;
    double dy = y - p.pose.position.y;
    double d2 = dx * dx + dy * dy;
    if (d2 < bestD2) { bestD2 = d2; best = i; }
  }
  m_cursor = best;

  // ── Arrival check ──
  const auto& finalPose = m_path.poses.back();
  double dToGoal = std::hypot(finalPose.pose.position.x - x,
                              finalPose.pose.position.y - y);
  bool atFinalCursor = (m_cursor >= m_path.poses.size() - 1);
  if (!m_arrived && (atFinalCursor || dToGoal < m_goalTolerance))
  {
    m_arrived = true;
    ROS_INFO("[rpp_guidance] arrived (cursor=%zu, dist=%.2f)",
             m_cursor, dToGoal);
  }
  if (m_arrived)
  {
    // if (!m_stopPublished)
    // {
    //   publishSpeedCourse(0.0, psi);
    //   // m_stopPublished = true;
    //   ROS_INFO("[rpp_guidance] arrived — emitting stop and ceasing publishes");
    // }
    publishSpeedCourse(0.0, psi);
    return;
    // stationKeep(x, y, psi, finalPose);
    // return;
  }

  // ── Direction from planner output ──
  double d_at_cursor = m_path.poses[m_cursor].pose.position.z;
  if (std::abs(d_at_cursor) < 0.5)
  {
    for (std::size_t i = m_cursor + 1; i < m_path.poses.size(); ++i)
    {
      double dd = m_path.poses[i].pose.position.z;
      if (std::abs(dd) > 0.5) { d_at_cursor = dd; break; }
    }
  }
  bool reverseMode = (d_at_cursor < 0.0);
  double sign = reverseMode ? -1.0 : 1.0;

  // ── Velocity-scaled lookahead and carrot ──
  double speed_for_lookahead;
  if (m_useDynamicWindow)
  {
    speed_for_lookahead = m_haveOdom
                              ? std::abs(m_currentVel.linear.x)
                              : std::abs(m_lastCommandedV);
  }
  else
  {
    speed_for_lookahead = std::abs(m_lastCommandedV);
  }
  double lookahead_dist = getLookAheadDistance(speed_for_lookahead);
  std::size_t carrot_idx = getCarrotIndex(lookahead_dist);
  geometry_msgs::PoseStamped carrot = m_path.poses[carrot_idx];

  // ── Pure-pursuit curvature toward the carrot ──
  double kappa_carrot = computePurePursuitCurvature(carrot, x, y, psi);

  // ── Path curvature (preview) for regulation ──
  // Use the maximum |kappa| between cursor and carrot so we slow down
  // entering tight curves rather than reacting to them.
  double kappa_preview = std::abs(kappa_carrot);
  // if (!m_curvatures.empty())
  // {
  //   std::size_t end = std::min(carrot_idx + 1, m_curvatures.size());
  //   for (std::size_t i = m_cursor; i < end; ++i)
  //   {
  //     double k = std::abs(static_cast<double>(m_curvatures[i]));
  //     if (k > kappa_preview) kappa_preview = k;
  //   }
  // }

  // ── Heading error (in body frame, used for regulation) ──
  double dxc = carrot.pose.position.x - x;
  double dyc = carrot.pose.position.y - y;
  double bearing = std::atan2(dyc, dxc);
  if (reverseMode) bearing += M_PI;
  double heading_err = bearing - psi;
  while (heading_err >  M_PI) heading_err -= 2.0 * M_PI;
  while (heading_err < -M_PI) heading_err += 2.0 * M_PI;

  // ── Apply regulation (the "soft cap" on linear velocity) ──
  double base_speed = m_maxLinearVel;
  double remaining  = remainingPathDistance();
  double regulated_v = applyConstraints(base_speed,
                                        kappa_preview,
                                        heading_err,
                                        remaining);
  if (reverseMode)
    regulated_v = -std::max(regulated_v, std::abs(m_minLinearVel));

  // ── Convert (regulated_v, kappa) → (v, w) ──
  // This is the upstream nav2 if-else, line for line.
  double v_cmd, w_cmd;
  if (!m_useDynamicWindow)
  {
    v_cmd = regulated_v;
    w_cmd = v_cmd * kappa_carrot;
  }
  else
  {
    geometry_msgs::Twist current = m_currentVel;
    if (!m_haveOdom)
    {
      current.linear.x  = m_lastCommandedV;
      current.angular.z = m_lastCommandedW;
    }
    const double dt = 1.0 / m_controlFrequency;
    std::tie(v_cmd, w_cmd) = dynamic_window_pure_pursuit::computeDynamicWindowVelocities(
        current,
        m_maxLinearVel,  m_minLinearVel,
        m_maxAngularVel, m_minAngularVel,
        m_maxLinearAccel, m_maxLinearDecel,
        m_maxAngularAccel, m_maxAngularDecel,
        regulated_v,
        kappa_carrot,
        sign,
        dt);
  }

  m_lastCommandedV = v_cmd;
  m_lastCommandedW = w_cmd;

  // ── (v, w) → (speed, course) ──
  // Project the heading forward over a short preview horizon so the
  // controller's heading PID has a real setpoint instead of an
  // instantaneous yaw rate.
  double dt_preview = std::max(0.2, std::min(m_lookaheadTime, 1.0));
  double course;
  if (reverseMode)
  {
    // For reverse motion, point the bow along the path tangent at the
    // carrot. The negative speed handles direction.
    course = tf2::getYaw(carrot.pose.orientation);
  }
  else
  {
    course = psi + w_cmd * dt_preview;
  }
  while (course >  M_PI) course -= 2.0 * M_PI;
  while (course < -M_PI) course += 2.0 * M_PI;

  publishSpeedCourse(v_cmd, course);
  publishCarrot(carrot);

  ROS_INFO_STREAM_THROTTLE(0.5,
      "[rpp_guidance] cursor=" << m_cursor << "/" << m_path.poses.size()
      << " carrot=" << carrot_idx
      << " L="      << lookahead_dist
      << " kp="     << kappa_preview
      << " v_reg="  << regulated_v
      << " v="      << v_cmd
      << " w="      << w_cmd
      << (m_useDynamicWindow ? " [DW]"  : " [RPP]")
      << (reverseMode        ? " [REV]" : " [FWD]"));
}

// ──────────────────────────────────────────────────────────────────────
// RPP building blocks
// ──────────────────────────────────────────────────────────────────────

double RegulatedPurePursuitGuidance::getLookAheadDistance(
    double current_speed) const
{
  if (!m_useVelocityScaledLookahead) return m_lookaheadDist;
  double L = current_speed * m_lookaheadTime;
  L = std::max(m_minLookaheadDist, std::min(m_maxLookaheadDist, L));
  return L;
}

std::size_t RegulatedPurePursuitGuidance::getCarrotIndex(
    double lookahead_dist) const
{
  if (m_path.poses.empty()) return 0;

  double accum = 0.0;
  for (std::size_t i = m_cursor + 1; i < m_path.poses.size(); ++i)
  {
    double dx = m_path.poses[i].pose.position.x
              - m_path.poses[i - 1].pose.position.x;
    double dy = m_path.poses[i].pose.position.y
              - m_path.poses[i - 1].pose.position.y;
    accum += std::hypot(dx, dy);
    if (accum >= lookahead_dist) return i;
  }
  return m_path.poses.size() - 1;
}

double RegulatedPurePursuitGuidance::computePurePursuitCurvature(
    const geometry_msgs::PoseStamped& carrot,
    double x, double y, double psi) const
{
  // Carrot in body frame
  double dx = carrot.pose.position.x - x;
  double dy = carrot.pose.position.y - y;
  double cs = std::cos(psi), sn = std::sin(psi);
  double xb =  cs * dx + sn * dy;
  double yb = -sn * dx + cs * dy;
  double d2 = xb * xb + yb * yb;
  if (d2 < 1e-6) return 0.0;
  // Standard pure-pursuit: positive yb (carrot to the left) → CCW turn.
  return 2.0 * yb / d2;
}

double RegulatedPurePursuitGuidance::applyConstraints(
    double base_linear_vel,
    double curvature,
    double heading_err,
    double dist_to_path_end) const
{
  double v = base_linear_vel;

  // 1. Curvature regulation: v scales with radius / R_min when the
  //    instantaneous radius drops below R_min. Floored at min_speed.
  if (m_useRegulatedLinearVelocityScaling && std::abs(curvature) > 1e-6)
  {
    double radius = 1.0 / std::abs(curvature);
    if (radius < m_regulatedLinearScalingMinRadius)
    {
      double scale = radius / m_regulatedLinearScalingMinRadius;
      double v_curv = std::max(m_regulatedLinearScalingMinSpeed,
                               base_linear_vel * scale);
      v = std::min(v, v_curv);
    }
  }

  // 2. Heading regulation: when the carrot is significantly off the
  //    bow, slow down before turning. Linear ramp from threshold to pi.
  if (m_useHeadingRegulation)
  {
    double he = std::abs(heading_err);
    if (he > m_headingScalingThreshold)
    {
      double range = M_PI - m_headingScalingThreshold;
      double scale = 1.0 - (he - m_headingScalingThreshold) / range;
      scale = std::max(0.1, scale);
      v = std::min(v, base_linear_vel * scale);
    }
  }

  // 3. Approach-velocity scaling near the goal.
  if (m_approachVelocityScalingDist > 0.0 &&
      dist_to_path_end < m_approachVelocityScalingDist)
  {
    double frac = dist_to_path_end / m_approachVelocityScalingDist;
    double v_app = m_minApproachLinearVelocity
                 + frac * (base_linear_vel - m_minApproachLinearVelocity);
    v = std::min(v, v_app);
  }

  return v;
}

double RegulatedPurePursuitGuidance::remainingPathDistance() const
{
  if (m_path.poses.size() < 2 || m_cursor + 1 >= m_path.poses.size())
    return 0.0;

  double accum = 0.0;
  for (std::size_t i = m_cursor + 1; i < m_path.poses.size(); ++i)
  {
    double dx = m_path.poses[i].pose.position.x
              - m_path.poses[i - 1].pose.position.x;
    double dy = m_path.poses[i].pose.position.y
              - m_path.poses[i - 1].pose.position.y;
    accum += std::hypot(dx, dy);
  }
  return accum;
}

// ──────────────────────────────────────────────────────────────────────
// Station-keeping (preserved from original guidance.cpp)
// ──────────────────────────────────────────────────────────────────────

// void RegulatedPurePursuitGuidance::stationKeep(
//     double x, double y, double psi,
//     const geometry_msgs::PoseStamped& goal)
// {
//   double dx = goal.pose.position.x - x;
//   double dy = goal.pose.position.y - y;
//   double dist = std::hypot(dx, dy);

//   const double deadzone = 0.3;
//   if (dist < deadzone)
//   {
//     publishSpeedCourse(0.0, tf2::getYaw(goal.pose.orientation));
//     return;
//   }

//   double bearing = std::atan2(dy, dx);
//   double he = bearing - psi;
//   while (he >  M_PI) he -= 2.0 * M_PI;
//   while (he < -M_PI) he += 2.0 * M_PI;
//   bool useReverse = std::abs(he) > M_PI_2;

//   double cmdHeading = useReverse ? bearing + M_PI : bearing;
//   while (cmdHeading >  M_PI) cmdHeading -= 2.0 * M_PI;
//   while (cmdHeading < -M_PI) cmdHeading += 2.0 * M_PI;

//   double creep = std::min(0.3, 0.4 * dist);
//   if (useReverse) creep = -creep;

//   publishSpeedCourse(creep, cmdHeading);
// }

// ──────────────────────────────────────────────────────────────────────
// Publishers
// ──────────────────────────────────────────────────────────────────────

void RegulatedPurePursuitGuidance::publishSpeedCourse(double speed,
                                                      double course)
{
  usv_msgs::SpeedCourse msg;
  msg.speed  = speed;
  msg.course = course;
  m_controllerPub.publish(msg);
}

void RegulatedPurePursuitGuidance::publishStop(double psi)
{
  publishSpeedCourse(0.0, psi);
}

void RegulatedPurePursuitGuidance::publishCarrot(
    const geometry_msgs::PoseStamped& carrot)
{
  geometry_msgs::PoseStamped msg = carrot;
  msg.header.stamp = ros::Time::now();
  if (msg.header.frame_id.empty()) msg.header.frame_id = "map";
  m_carrotPub.publish(msg);
}

}  // namespace blueboat_coverage
