#ifndef BLUEBOAT_CONTROL_H
#define BLUEBOAT_CONTROL_H

#include <ros/ros.h>

#include <geometry_msgs/Vector3Stamped.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <usv_msgs/SpeedCourse.h>

#include <eigen3/Eigen/Core>

class BlueboatController
{
public:
  BlueboatController();

private:
  double calculateSurgeForce(double deltaTime, double u);
  double calculateYawMoment(double deltaTime, double psi, double r);
  Eigen::Vector2d thrustAllocation(Eigen::Vector3d tau_d);
  void inputCallback(const usv_msgs::SpeedCourse& msg);
  void odomCallback(const nav_msgs::Odometry& msg);
  double getYaw();
  void loadGains();

  ros::NodeHandle nh_private;

  // Heading controller
  double Kp_psi;
  double Ki_psi;
  double Kd_psi;
  double mass_psi;
  double damp_psi;

  // Surge speed controller (single set of gains; sign of u_d carries
  // direction. No separate reverse PID.)
  double Kp_u;
  double Ki_u;
  double mass_u;
  double damp_u;

  // Sensor data
  double u   = 0.0;
  double psi = 0.0;
  double r   = 0.0;

  // Desired values
  double u_d   = 0.0;
  double psi_d = 0.0;

  // Thruster configuration matrix
  Eigen::MatrixXd T;

  // Publishers
  ros::Publisher m_leftPub;
  ros::Publisher m_rightPub;
  ros::Publisher m_debugSpeedActual;
  ros::Publisher m_debugSpeedDesired;
  ros::Publisher m_debugYawActual;
  ros::Publisher m_debugYawDesired;
  ros::Publisher m_debugThrustL;
  ros::Publisher m_debugThrustR;
};

#endif
