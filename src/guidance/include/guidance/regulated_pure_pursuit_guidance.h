#ifndef BLUEBOAT_GUIDANCE_REGULATED_PURE_PURSUIT_GUIDANCE_H_
#define BLUEBOAT_GUIDANCE_REGULATED_PURE_PURSUIT_GUIDANCE_H_

/**
 * regulated_pure_pursuit_guidance.h
 *
 * USV port of nav2_regulated_pure_pursuit_controller. Single class that
 * implements RPP and exposes a `use_dynamic_window` rosparam to optionally
 * enable Ohnishi's Dynamic Window Pure Pursuit step at the end of the
 * compute cycle. This mirrors nav2's design: in upstream
 * nav2_regulated_pure_pursuit_controller.cpp the same controller class
 * branches on `use_dynamic_window` between "omega = v * kappa" and the
 * dynamic-window solver. We do the same.
 *
 * Adaptations from the upstream nav2 code:
 *   - Output is usv_msgs/SpeedCourse (speed, course) instead of
 *     geometry_msgs/Twist (v, omega). Course is computed by projecting
 *     the implied yaw rate forward over a short preview horizon, so the
 *     existing controller's heading PID has a real setpoint to track.
 *   - Cursor-based forward-only path consumption is preserved from the
 *     original blueboat guidance so HC-RS / Reeds-Shepp paths with cusps
 *     don't trip up the carrot search.
 *   - Direction (forward/reverse) is read from pose.position.z, which
 *     carries the steering library's d ∈ {-1, 0, +1}. No heading-error
 *     heuristic.
 *   - Costmap obstacle/collision checking is omitted; that belongs in the
 *     local planner (the VFH node), not in guidance.
 *   - rotateToHeading / shouldRotateToGoalHeading are not implemented:
 *     a USV cannot turn in place, so the upstream "spin to heading" mode
 *     would just stall the boat. Station-keeping at goal is handled by
 *     stationKeep() instead.
 *
 * I/O contract:
 *   Subscribes:
 *     ~path_topic       (nav_msgs/Path,        default "planned_path")
 *     ~curvature_topic  (Float32MultiArray,    default "planned_curvatures")
 *     ~odom_topic       (nav_msgs/Odometry,    default "/odometry/filtered")
 *                         — only used when use_dynamic_window=true; falls
 *                           back to last-commanded velocity if missing.
 *   Publishes:
 *     speed_heading     (usv_msgs/SpeedCourse) → controller
 *     ~carrot           (geometry_msgs/PoseStamped) → RViz
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32MultiArray.h>

#include <vector>
#include <cstddef>

namespace blueboat_coverage
{

class RegulatedPurePursuitGuidance
{
public:
  RegulatedPurePursuitGuidance();
  ~RegulatedPurePursuitGuidance();

private:
  // ── Callbacks ──
  void newPath(const nav_msgs::Path& path);
  void newCurvatures(const std_msgs::Float32MultiArray& msg);
  void newOdom(const nav_msgs::Odometry& msg);

  // ── Top-level step ──
  void followPath(double x, double y, double psi);

  // ── RPP building blocks (named after nav2 equivalents) ──
  /** Velocity-scaled lookahead distance, clamped to [min, max].
   *  Falls back to a fixed value if velocity scaling is disabled. */
  double getLookAheadDistance(double current_speed) const;

  /** Find the path pose at lookahead_dist arc-length ahead of m_cursor. */
  std::size_t getCarrotIndex(double lookahead_dist) const;

  /** kappa = 2 * y_body / d^2 toward the carrot. Standard pure pursuit. */
  double computePurePursuitCurvature(const geometry_msgs::PoseStamped& carrot,
                                     double x, double y, double psi) const;

  /** Apply curvature, heading-error and approach-velocity regulation.
   *  Mirrors nav2 RPP::applyConstraints. */
  double applyConstraints(double base_linear_vel,
                          double curvature,
                          double heading_err,
                          double dist_to_path_end) const;

  double remainingPathDistance() const;

  //void stationKeep(double x, double y, double psi, const geometry_msgs::PoseStamped& goal);

  void publishSpeedCourse(double speed, double course);
  void publishStop(double psi);
  void publishCarrot(const geometry_msgs::PoseStamped& carrot);

  // ── State ──
  nav_msgs::Path     m_path;
  std::vector<float> m_curvatures;
  std::size_t        m_cursor   = 0;
  bool               m_arrived  = false;
  //bool 	     m_stopPublished = false;

  // Measured velocity in body frame (from /odometry/filtered).
  // Only used when m_useDynamicWindow=true.
  geometry_msgs::Twist m_currentVel;
  bool                 m_haveOdom        = false;
  double               m_lastCommandedV  = 0.0;
  double               m_lastCommandedW  = 0.0;

  // ── Pubs / subs ──
  ros::Publisher  m_controllerPub;
  ros::Publisher  m_carrotPub;
  ros::Subscriber m_pathSub;
  ros::Subscriber m_kappaSub;
  ros::Subscriber m_odomSub;

  // ── Parameters (names mirror nav2 RPP wherever sensible) ──
  // Mode flag — the one that flips DWPP on/off, exactly like upstream.
  bool   m_useDynamicWindow;

  // Lookahead
  bool   m_useVelocityScaledLookahead;
  double m_lookaheadDist;
  double m_minLookaheadDist;
  double m_maxLookaheadDist;
  double m_lookaheadTime;

  // Curvature regulation
  bool   m_useRegulatedLinearVelocityScaling;
  double m_regulatedLinearScalingMinRadius;
  double m_regulatedLinearScalingMinSpeed;

  // Heading regulation
  bool   m_useHeadingRegulation;
  double m_headingScalingThreshold;

  // Approach velocity scaling
  double m_approachVelocityScalingDist;
  double m_minApproachLinearVelocity;

  // Speed limits
  double m_maxLinearVel;
  double m_minLinearVel;
  double m_maxAngularVel;   // used only when use_dynamic_window=true
  double m_minAngularVel;   // used only when use_dynamic_window=true

  // Acceleration limits — used only when use_dynamic_window=true
  double m_maxLinearAccel;
  double m_maxLinearDecel;   // negative
  double m_maxAngularAccel;
  double m_maxAngularDecel;  // negative

  // Arrival / cursor
  double m_goalTolerance;
  int    m_searchWindow;

  // Loop
  double m_controlFrequency;
};

}  // namespace blueboat_coverage

#endif  // BLUEBOAT_GUIDANCE_REGULATED_PURE_PURSUIT_GUIDANCE_H_
