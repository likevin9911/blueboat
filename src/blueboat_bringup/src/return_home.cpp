/**
 * return_home.cpp
 *
 * Simple return-to-home state machine for a USV.
 *
 * State flow:
 *   IDLE → GO_TO_WAYPOINT → UTURN → RETURN_HOME → DONE
 *
 * 1. Records home pose (position + heading) from TF at startup.
 * 2. Generates a Dubins path from home to the waypoint and publishes it
 *    on "simple_dubins_path" for the Guidance node to follow.
 * 3. On arrival at the waypoint (position + heading aligned), commands
 *    a 180° in-place spin (U-turn).
 * 4. Generates a Dubins path from the waypoint back home and publishes it.
 * 5. On arrival home (position + heading aligned), publishes an empty path
 *    so Guidance stops.
 *
 * Arrival is detected by pose alignment: the boat must be within
 * position_tolerance of the goal AND heading must be within
 * heading_tolerance of the target heading. This ensures the boat
 * actually reaches the final Dubins pose rather than triggering
 * early inside a radius.
 *
 * Publishes:
 *   nav_msgs/Path           "simple_dubins_path"  (Guidance consumes this)
 *   usv_msgs/SpeedCourse    "speed_heading"        (direct cmd during U-turn)
 *   std_msgs/Bool           "return_home/active"
 *
 * Parameters:
 *   ~waypoint_x           double  (required)
 *   ~waypoint_y           double  (required)
 *   ~home_x               double  (fallback if TF unavailable)
 *   ~home_y               double  (fallback if TF unavailable)
 *   ~home_heading         double  (fallback, radians)
 *   ~goal_tolerance       double  radius for waypoint arrival → U-turn, m (default 1.5)
 *   ~position_tolerance   double  how close to home position, m           (default 1.5)
 *   ~heading_tolerance    double  how close to home heading, rad          (default 0.25 ~14°)
 *   ~min_turn_radius      double  Dubins turning radius, m               (default 2.0)
 *   ~map_frame            string  (default "map")
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <usv_msgs/SpeedCourse.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

#include <cmath>
#include <vector>
#include <limits>

namespace blueboat_coverage
{

// ─────────────────────────────────────────────────────────────────────────
// Wrap angle to [-pi, pi]
// ─────────────────────────────────────────────────────────────────────────
static double wrapToPi(double a)
{
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

// ─────────────────────────────────────────────────────────────────────────
// Dubins path builder — evaluates LSL, RSR, LSR, RSL and picks shortest
// ─────────────────────────────────────────────────────────────────────────
struct DubinsPose { double x, y, psi; };

static void sampleArc(std::vector<geometry_msgs::PoseStamped>& poses,
                       const std_msgs::Header& header,
                       double cx, double cy, double R,
                       double a0, double a1, int dir, double step)
{
  double dpsi = a1 - a0;
  if (dir > 0 && dpsi < 0) dpsi += 2.0 * M_PI;
  if (dir < 0 && dpsi > 0) dpsi -= 2.0 * M_PI;

  double arc_len = std::fabs(dpsi) * R;
  int n = std::max(2, static_cast<int>(arc_len / step));

  for (int i = 0; i <= n; ++i)
  {
    double a = a0 + dpsi * static_cast<double>(i) / n;
    geometry_msgs::PoseStamped ps;
    ps.header = header;
    ps.pose.position.x = cx + R * std::cos(a);
    ps.pose.position.y = cy + R * std::sin(a);
    double tangent = (dir > 0) ? (a + M_PI_2) : (a - M_PI_2);
    tf2::Quaternion q;
    q.setRPY(0, 0, tangent);
    ps.pose.orientation = tf2::toMsg(q);
    poses.push_back(ps);
  }
}

static void sampleLine(std::vector<geometry_msgs::PoseStamped>& poses,
                        const std_msgs::Header& header,
                        double x1, double y1, double x2, double y2,
                        double step)
{
  double dist = std::hypot(x2 - x1, y2 - y1);
  if (dist < 1e-6) return;

  double psi = std::atan2(y2 - y1, x2 - x1);
  int n = std::max(2, static_cast<int>(dist / step));

  for (int i = 0; i <= n; ++i)
  {
    double t = static_cast<double>(i) / n;
    geometry_msgs::PoseStamped ps;
    ps.header = header;
    ps.pose.position.x = x1 + t * (x2 - x1);
    ps.pose.position.y = y1 + t * (y2 - y1);
    tf2::Quaternion q;
    q.setRPY(0, 0, psi);
    ps.pose.orientation = tf2::toMsg(q);
    poses.push_back(ps);
  }
}

static double arcLength(double a0, double a1, int dir, double R)
{
  double dpsi = a1 - a0;
  if (dir > 0 && dpsi < 0) dpsi += 2.0 * M_PI;
  if (dir < 0 && dpsi > 0) dpsi -= 2.0 * M_PI;
  return std::fabs(dpsi) * R;
}

struct CSCResult
{
  double c1x, c1y;
  double c2x, c2y;
  double tp1x, tp1y;
  double tp2x, tp2y;
  double a_start1, a_end1;
  double a_start2, a_end2;
  int d1, d2;
  double length;
};

static CSCResult tryCSC(const DubinsPose& from, const DubinsPose& to,
                         double R, int d1, int d2)
{
  CSCResult res;
  res.d1 = d1;
  res.d2 = d2;
  res.length = std::numeric_limits<double>::infinity();

  double offset1 = (d1 > 0) ? M_PI_2 : -M_PI_2;
  double offset2 = (d2 > 0) ? M_PI_2 : -M_PI_2;

  res.c1x = from.x + R * std::cos(from.psi + offset1);
  res.c1y = from.y + R * std::sin(from.psi + offset1);
  res.c2x = to.x   + R * std::cos(to.psi   + offset2);
  res.c2y = to.y   + R * std::sin(to.psi   + offset2);

  double dx = res.c2x - res.c1x;
  double dy = res.c2y - res.c1y;
  double D  = std::hypot(dx, dy);

  if (D < 1e-4) return res;

  double theta = std::atan2(dy, dx);

  if (d1 == d2)
  {
    double tangent_offset = (d1 > 0) ? -M_PI_2 : M_PI_2;
    res.tp1x = res.c1x + R * std::cos(theta + tangent_offset);
    res.tp1y = res.c1y + R * std::sin(theta + tangent_offset);
    res.tp2x = res.c2x + R * std::cos(theta + tangent_offset);
    res.tp2y = res.c2y + R * std::sin(theta + tangent_offset);
  }
  else
  {
    if (D < 2.0 * R) return res;

    double alpha = std::acos(2.0 * R / D);
    double tangent_angle;

    if (d1 > 0)
      tangent_angle = theta - alpha;
    else
      tangent_angle = theta + alpha;

    double tp_offset1 = (d1 > 0) ? (tangent_angle - M_PI_2) : (tangent_angle + M_PI_2);
    res.tp1x = res.c1x + R * std::cos(tp_offset1);
    res.tp1y = res.c1y + R * std::sin(tp_offset1);

    double tp_offset2 = (d2 > 0) ? (tangent_angle + M_PI + M_PI_2) : (tangent_angle + M_PI - M_PI_2);
    res.tp2x = res.c2x + R * std::cos(tp_offset2);
    res.tp2y = res.c2y + R * std::sin(tp_offset2);
  }

  res.a_start1 = std::atan2(from.y - res.c1y, from.x - res.c1x);
  res.a_end1   = std::atan2(res.tp1y - res.c1y, res.tp1x - res.c1x);
  res.a_start2 = std::atan2(res.tp2y - res.c2y, res.tp2x - res.c2x);
  res.a_end2   = std::atan2(to.y - res.c2y, to.x - res.c2x);

  double len1 = arcLength(res.a_start1, res.a_end1, d1, R);
  double len2 = std::hypot(res.tp2x - res.tp1x, res.tp2y - res.tp1y);
  double len3 = arcLength(res.a_start2, res.a_end2, d2, R);

  res.length = len1 + len2 + len3;
  return res;
}

static nav_msgs::Path buildDubinsPath(const DubinsPose& from,
                                       const DubinsPose& to,
                                       double R,
                                       const std::string& frame_id,
                                       double step = 0.3)
{
  nav_msgs::Path path;
  path.header.stamp    = ros::Time::now();
  path.header.frame_id = frame_id;

  CSCResult candidates[4] = {
    tryCSC(from, to, R, +1, +1),
    tryCSC(from, to, R, -1, -1),
    tryCSC(from, to, R, +1, -1),
    tryCSC(from, to, R, -1, +1),
  };

  CSCResult* best = &candidates[0];
  for (int i = 1; i < 4; ++i)
    if (candidates[i].length < best->length)
      best = &candidates[i];

  if (std::isinf(best->length))
  {
    ROS_WARN("[ReturnHome] No valid Dubins path — using straight line");
    sampleLine(path.poses, path.header, from.x, from.y, to.x, to.y, step);
    return path;
  }

  sampleArc(path.poses, path.header,
            best->c1x, best->c1y, R,
            best->a_start1, best->a_end1, best->d1, step);

  sampleLine(path.poses, path.header,
             best->tp1x, best->tp1y, best->tp2x, best->tp2y, step);

  sampleArc(path.poses, path.header,
            best->c2x, best->c2y, R,
            best->a_start2, best->a_end2, best->d2, step);

  ROS_INFO("[ReturnHome] Dubins path: %s%s  length=%.1f m  %zu poses",
           (best->d1 > 0 ? "L" : "R"), (best->d2 > 0 ? "L" : "R"),
           best->length, path.poses.size());

  return path;
}

// ─────────────────────────────────────────────────────────────────────────
// State machine
// ─────────────────────────────────────────────────────────────────────────
enum class State { IDLE, GO_TO_WAYPOINT, UTURN, RETURN_HOME, DONE };

class ReturnHome
{
public:
  ReturnHome()
    : m_tfListener(m_tfBuffer)
    , m_state(State::IDLE)
    , m_lastReplan(0.0)
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    m_waypointX     = nhP.param("waypoint_x",          0.0);
    m_waypointY     = nhP.param("waypoint_y",          0.0);
    m_goalTol       = nhP.param("goal_tolerance",      1.5);   // radius for waypoint arrival → U-turn
    m_posTol        = nhP.param("position_tolerance",  1.5);   // position for home arrival
    m_hdgTol        = nhP.param("heading_tolerance",   0.25);  // heading for home arrival (~14°)
    m_turnRadius    = nhP.param("min_turn_radius",     2.0);
    m_frameId       = nhP.param<std::string>("map_frame", "map");

    recordHome(nhP);

    m_pathPub   = nh.advertise<nav_msgs::Path>("simple_dubins_path", 10, true);
    m_statusPub = nh.advertise<std_msgs::Bool>("return_home/active", 10, true);
    m_cmdPub    = nh.advertise<usv_msgs::SpeedCourse>("speed_heading", 10);

    ROS_INFO("[ReturnHome] Ready. waypoint=(%.1f, %.1f)  home=(%.1f, %.1f) hdg=%.1f deg  R=%.1f m  "
             "goal_tol=%.2f m  home_pos_tol=%.2f m  home_hdg_tol=%.1f deg",
             m_waypointX, m_waypointY, m_homeX, m_homeY,
             m_homePsi * 180.0 / M_PI, m_turnRadius,
             m_goalTol, m_posTol, m_hdgTol * 180.0 / M_PI);

    ros::Rate rate(5.0);
    while (nh.ok())
    {
      update();
      ros::spinOnce();
      rate.sleep();
    }
  }

private:
  tf2_ros::Buffer            m_tfBuffer;
  tf2_ros::TransformListener m_tfListener;
  ros::Publisher             m_pathPub;
  ros::Publisher             m_statusPub;
  ros::Publisher             m_cmdPub;

  double      m_waypointX, m_waypointY;
  double      m_homeX, m_homeY, m_homePsi;
  double      m_goalTol;          // radius-only check for waypoint → triggers U-turn
  double      m_posTol;           // position check for home arrival
  double      m_hdgTol;           // heading  check for home arrival
  double      m_turnRadius;
  std::string m_frameId;

  State     m_state;
  double    m_outboundPsi;
  double    m_returnPsi;
  double    m_uturnTargetPsi;
  ros::Time m_uturnStartTime;
  ros::Time m_lastReplan;

  // ─────────────────────────────────────────────────────────────────────
  void recordHome(ros::NodeHandle& nhP)
  {
    for (int i = 0; i < 10; ++i)
    {
      try
      {
        auto tf = m_tfBuffer.lookupTransform(m_frameId, "base_link",
                                              ros::Time(0), ros::Duration(1.0));
        m_homeX   = tf.transform.translation.x;
        m_homeY   = tf.transform.translation.y;
        m_homePsi = tf2::getYaw(tf.transform.rotation);
        ROS_INFO("[ReturnHome] Home from TF: (%.2f, %.2f) heading=%.1f deg",
                 m_homeX, m_homeY, m_homePsi * 180.0 / M_PI);
        return;
      }
      catch (tf2::TransformException&) { ros::Duration(0.5).sleep(); }
    }

    m_homeX   = nhP.param("home_x",       0.0);
    m_homeY   = nhP.param("home_y",       0.0);
    m_homePsi = nhP.param("home_heading", 0.0);
    ROS_WARN("[ReturnHome] TF unavailable — param home (%.2f, %.2f)", m_homeX, m_homeY);
  }

  // ─────────────────────────────────────────────────────────────────────
  bool getRobotPose(double& x, double& y, double& psi)
  {
    try
    {
      auto tf = m_tfBuffer.lookupTransform(m_frameId, "base_link",
                                            ros::Time(0), ros::Duration(0.1));
      x   = tf.transform.translation.x;
      y   = tf.transform.translation.y;
      psi = tf2::getYaw(tf.transform.rotation);
      return true;
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN_THROTTLE(2.0, "[ReturnHome] TF error: %s", ex.what());
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────
  // Pose-based arrival check: position AND heading must both be within
  // tolerance. This replaces the old radius-only goal_tolerance.
  // ─────────────────────────────────────────────────────────────────────
  bool poseReached(double x, double y, double psi,
                   double goal_x, double goal_y, double goal_psi)
  {
    double dist    = std::hypot(x - goal_x, y - goal_y);
    double hdg_err = std::fabs(wrapToPi(psi - goal_psi));
    return (dist < m_posTol) && (hdg_err < m_hdgTol);
  }

  // ─────────────────────────────────────────────────────────────────────
  void publishPath(double from_x, double from_y, double from_psi,
                   double to_x, double to_y, double to_psi)
  {
    DubinsPose from{from_x, from_y, from_psi};
    DubinsPose to{to_x, to_y, to_psi};
    nav_msgs::Path path = buildDubinsPath(from, to, m_turnRadius, m_frameId);
    m_pathPub.publish(path);
  }

  // ─────────────────────────────────────────────────────────────────────
  void stopBoat()
  {
    nav_msgs::Path empty;
    empty.header.stamp    = ros::Time::now();
    empty.header.frame_id = m_frameId;
    m_pathPub.publish(empty);
  }

  // ─────────────────────────────────────────────────────────────────────
  void update()
  {
    double x, y, psi;
    if (!getRobotPose(x, y, psi)) return;

    std_msgs::Bool status;
    status.data = (m_state != State::IDLE && m_state != State::DONE);
    m_statusPub.publish(status);

    switch (m_state)
    {
      // ── Start: lock headings, publish outbound path ─────────────────
      case State::IDLE:
      {
        m_outboundPsi    = std::atan2(m_waypointY - m_homeY, m_waypointX - m_homeX);
        m_returnPsi      = wrapToPi(m_outboundPsi + M_PI);
        m_uturnTargetPsi = m_returnPsi;

        // Path from exact home pose to exact waypoint pose
        publishPath(m_homeX, m_homeY, m_outboundPsi,
                    m_waypointX, m_waypointY, m_outboundPsi);

        ROS_INFO("[ReturnHome] IDLE -> GO_TO_WAYPOINT  wp=(%.1f,%.1f)  "
                 "outbound=%.1f deg  return=%.1f deg",
                 m_waypointX, m_waypointY,
                 m_outboundPsi * 180.0 / M_PI,
                 m_returnPsi   * 180.0 / M_PI);
        m_state = State::GO_TO_WAYPOINT;
        break;
      }

      // ── Leg 1: follow Dubins path to waypoint ──────────────────────
      case State::GO_TO_WAYPOINT:
      {
        double dist = std::hypot(x - m_waypointX, y - m_waypointY);

        if (dist < m_goalTol)
        {
          // Within radius — trigger U-turn
          m_uturnStartTime = ros::Time::now();
          m_state = State::UTURN;
          ROS_INFO("[ReturnHome] Waypoint reached (%.2f m). U-turn to %.1f deg",
                   dist, m_uturnTargetPsi * 180.0 / M_PI);
        }
        else if ((ros::Time::now() - m_lastReplan).toSec() > 3.0)
        {
          // Replan anchored to exact endpoints — Guidance snaps to
          // nearest point and drives through the waypoint
          publishPath(m_homeX, m_homeY, m_outboundPsi,
                      m_waypointX, m_waypointY, m_outboundPsi);
          m_lastReplan = ros::Time::now();
        }
        break;
      }

      // ── U-turn: spin 180° in place ─────────────────────────────────
      case State::UTURN:
      {
        usv_msgs::SpeedCourse cmd;
        cmd.speed  = 0.0;
        cmd.course = m_uturnTargetPsi;
        m_cmdPub.publish(cmd);

        double err     = std::fabs(wrapToPi(m_uturnTargetPsi - psi));
        double elapsed = (ros::Time::now() - m_uturnStartTime).toSec();

        if (err < 0.15 || elapsed > 20.0)   // ~8.5° tolerance or timeout
        {
          ROS_INFO("[ReturnHome] U-turn done (err=%.1f deg, %.1fs). Heading home.",
                   err * 180.0 / M_PI, elapsed);

          // Return path from exact waypoint to exact home
          publishPath(m_waypointX, m_waypointY, m_returnPsi,
                      m_homeX, m_homeY, m_returnPsi);
          m_lastReplan = ros::Time::now();
          m_state = State::RETURN_HOME;
        }
        break;
      }

      // ── Leg 2: follow Dubins path home ─────────────────────────────
      case State::RETURN_HOME:
      {
        if (poseReached(x, y, psi, m_homeX, m_homeY, m_returnPsi))
        {
          ROS_INFO("[ReturnHome] Home pose reached (%.2f m, hdg=%.1f deg). Done.",
                   std::hypot(x - m_homeX, y - m_homeY),
                   std::fabs(wrapToPi(psi - m_returnPsi)) * 180.0 / M_PI);
          stopBoat();
          m_state = State::DONE;
        }
        else if ((ros::Time::now() - m_lastReplan).toSec() > 3.0)
        {
          publishPath(m_waypointX, m_waypointY, m_returnPsi,
                      m_homeX, m_homeY, m_returnPsi);
          m_lastReplan = ros::Time::now();
        }
        break;
      }

      case State::DONE:
        break;
    }
  }
};

} // namespace blueboat_coverage

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "return_home");
  blueboat_coverage::ReturnHome node;
  return 0;
}