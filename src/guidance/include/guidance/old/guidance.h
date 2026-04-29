#ifndef GUIDANCE_H_
#define GUIDANCE_H_

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32MultiArray.h>

#include <vector>

namespace blueboat_coverage
{

class Guidance
{
public:
  Guidance();
  ~Guidance();

private:
  // Lookahead-distance shaping constants. Tunable; left as compile-time
  // since they were that way in the previous version.
  static constexpr double delta_max = 4.0;
  static constexpr double delta_min = 1.0;
  static constexpr double delta_k   = 0.5;

  // Callbacks
  void newPath(const nav_msgs::Path& path);
  void newCurvatures(const std_msgs::Float32MultiArray& msg);

  // Main loop work
  void followPath(double x, double y, double psi);
  void publishStop(double psi);

  // ROS I/O
  ros::Publisher m_controllerPub;

  // Cached planner outputs
  nav_msgs::Path     m_path;
  std::vector<float> m_curvatures;

  // Cursor into m_path. Forward-only.
  size_t m_cursor{0};

  // Tunable params (read from rosparam in ctor)
  double m_maxSpeed;
  double m_maxSpeedTurn;
  double m_minSpeed;
  double m_kappaTurnThreshold;
  int    m_searchWindow;
};

} // namespace blueboat_coverage

#endif
