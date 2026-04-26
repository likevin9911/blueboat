#ifndef BLUEBOAT_GUIDANCE_H_
#define BLUEBOAT_GUIDANCE_H_

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32MultiArray.h>
#include <ros/ros.h>

#include <cstddef>
#include <vector>

namespace blueboat_coverage
{

class Guidance
{
public:
  Guidance();
  ~Guidance();

private:
  void newWaypoint(const geometry_msgs::PoseStamped& waypoint);
  void newPath(const nav_msgs::Path& path);
  void newCurvatures(const std_msgs::Float32MultiArray& msg);
  void followPath(double x, double y, double psi);
  void publishStop(double psi);
  void stationKeep(double x, double y, double psi,
                   const geometry_msgs::PoseStamped& goal);
  double dist(double x0, double y0, double x1, double y1) const;

  nav_msgs::Path     m_path;
  std::vector<float> m_curvatures;

  ros::Publisher m_controllerPub;

  // Forward-only cursor into m_path.poses. Resets to 0 on new path.
  std::size_t m_cursor{0};
  bool        m_arrived{false};
  int         m_searchWindow{50};
  double      m_goalTolerance{0.5};

  // lookahead distance
  double DELTA = 0.5;
  double delta_max = 4.0;
  double delta_min = 1.0;
  double delta_k   = 1.0;

  // circle of acceptance (legacy, kept for compatibility)
  double R = 1.0;

  double m_maxSpeed;
  double m_maxSpeedTurn;
  double m_minSpeed;

  double m_kappaTurnThreshold;
};

} // namespace blueboat_coverage

#endif
