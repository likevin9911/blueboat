/**
 * controller.cpp  (heuristics-free version)
 *
 * Single PID per axis (surge and yaw). Forward/reverse direction is
 * handled by the sign of u_d coming from guidance — there is no separate
 * gain set and no logic that decides "we should drive backwards." If the
 * guidance node says u_d < 0, we track u_d < 0 with the same gains used
 * for u_d > 0.
 *
 * The integrator reset on direction change is retained — it's a wind-up
 * fix, not a behavior heuristic. When commanded speed flips sign, the
 * accumulated error term from the prior direction is no longer relevant
 * to the new direction and would actively oppose the new command.
 */

#include <blueboat_control/controller.h>

#include <std_msgs/Float32.h>

#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>

#include <eigen3/Eigen/QR>

#include <cmath>

BlueboatController::BlueboatController() : T(3, 2), nh_private("~")
{
  ros::NodeHandle nh;

  nh_private.param("Kp_psi",   Kp_psi,   1.0);
  nh_private.param("Ki_psi",   Ki_psi,   0.0);
  nh_private.param("Kd_psi",   Kd_psi,   0.0);
  nh_private.param("mass_psi", mass_psi, 9.0);
  nh_private.param("damp_psi", damp_psi, 20.0);

  nh_private.param("Kp_u",     Kp_u,     2.0);
  nh_private.param("Ki_u",     Ki_u,     1.0);
  nh_private.param("mass_u",   mass_u,   19.28);
  nh_private.param("damp_u",   damp_u,   20.0);

  m_leftPub  = nh.advertise<std_msgs::Float32>("left_thrust_cmd",  10);
  m_rightPub = nh.advertise<std_msgs::Float32>("right_thrust_cmd", 10);

  m_debugSpeedActual  = nh.advertise<std_msgs::Float32>("debug/speed_actual",  10);
  m_debugSpeedDesired = nh.advertise<std_msgs::Float32>("debug/speed_desired", 10);
  m_debugYawActual    = nh.advertise<std_msgs::Float32>("debug/yaw_actual",    10);
  m_debugYawDesired   = nh.advertise<std_msgs::Float32>("debug/yaw_desired",   10);
  m_debugThrustL      = nh.advertise<std_msgs::Float32>("debug/thrust_left",   10);
  m_debugThrustR      = nh.advertise<std_msgs::Float32>("debug/thrust_right",  10);

  ros::Subscriber sub     = nh.subscribe("speed_heading",      10,
                                         &BlueboatController::inputCallback, this);
  ros::Subscriber subOdom = nh.subscribe("/odometry/filtered", 10,
                                         &BlueboatController::odomCallback,  this);

  // Thruster configuration matrix (twin thrusters, lateral spacing 0.39 m)
  T << 50, 50,
       0,  0,
       -0.52 * 50, 0.52 * 50;

  const double frequency = 10.0;
  const double deltaTime = 1.0 / frequency;
  ros::Rate rate(frequency);

  while (nh.ok())
  {
    loadGains();

    double psi_slam = getYaw();
    double tauSurge = calculateSurgeForce(deltaTime, u);
    double tauYaw   = calculateYawMoment(deltaTime, psi_slam, r);

    Eigen::Vector2d cmdThrust = thrustAllocation({tauSurge, 0.0, tauYaw});

    std_msgs::Float32 left;
    left.data = static_cast<float>(cmdThrust[0]);
    std_msgs::Float32 right;
    right.data = static_cast<float>(cmdThrust[1]);
    m_leftPub.publish(left);
    m_rightPub.publish(right);

    std_msgs::Float32 dbg;
    dbg.data = static_cast<float>(u);            m_debugSpeedActual.publish(dbg);
    dbg.data = static_cast<float>(u_d);          m_debugSpeedDesired.publish(dbg);
    dbg.data = static_cast<float>(psi_slam);     m_debugYawActual.publish(dbg);
    dbg.data = static_cast<float>(psi_d);        m_debugYawDesired.publish(dbg);
    dbg.data = static_cast<float>(cmdThrust[0]); m_debugThrustL.publish(dbg);
    dbg.data = static_cast<float>(cmdThrust[1]); m_debugThrustR.publish(dbg);

    ros::spinOnce();
    rate.sleep();
  }
}

void BlueboatController::loadGains()
{
  nh_private.getParam("Kp_psi",   Kp_psi);
  nh_private.getParam("Ki_psi",   Ki_psi);
  nh_private.getParam("Kd_psi",   Kd_psi);
  nh_private.getParam("mass_psi", mass_psi);
  nh_private.getParam("damp_psi", damp_psi);

  nh_private.getParam("Kp_u",     Kp_u);
  nh_private.getParam("Ki_u",     Ki_u);
  nh_private.getParam("mass_u",   mass_u);
  nh_private.getParam("damp_u",   damp_u);
}

double BlueboatController::calculateSurgeForce(double deltaTime, double u)
{
  static double integralTerm = 0.0;
  static double prev_u_d     = 0.0;

  double u_d_dot = 0.0;
  double u_tilde = u - u_d;

  // Anti-windup: clear the integrator when the commanded direction flips.
  // This is a stability fix, not a behavior heuristic — error accumulated
  // in one direction is not meaningful in the other.
  if ((prev_u_d >= 0.0 && u_d < 0.0) ||
      (prev_u_d <  0.0 && u_d >= 0.0))
  {
    integralTerm = 0.0;
    ROS_INFO("Surge integral reset: direction change");
  }
  prev_u_d = u_d;

  integralTerm += u_tilde * deltaTime;

  return mass_u * (u_d_dot - Kp_u * u_tilde - Ki_u * integralTerm)
         + damp_u * u;
}

double BlueboatController::calculateYawMoment(double deltaTime,
                                              double psi_slam,
                                              double r)
{
  static double integralTerm = 0.0;

  double r_d_dot   = 0.0;
  double r_tilde   = 0.0;
  double psi_tilde = psi_slam - psi_d;
  while (psi_tilde >  M_PI) psi_tilde -= 2 * M_PI;
  while (psi_tilde < -M_PI) psi_tilde += 2 * M_PI;

  // Integrate with clamp to prevent windup.
  integralTerm += psi_tilde * deltaTime;
  const double iMax = 2.0;  // tune: rad·s
  if      (integralTerm >  iMax) integralTerm =  iMax;
  else if (integralTerm < -iMax) integralTerm = -iMax;

  return mass_psi * (r_d_dot - Kd_psi * r_tilde
                              - Kp_psi * psi_tilde
                              - Ki_psi * integralTerm)
         - damp_psi * r;
}

Eigen::Vector2d BlueboatController::thrustAllocation(Eigen::Vector3d tau_d)
{
  static bool initialized = false;
  static Eigen::MatrixXd pinv(3, 2);
  if (!initialized)
  {
    initialized = true;
    pinv = T.completeOrthogonalDecomposition().pseudoInverse();
  }

  Eigen::Vector2d u = pinv * tau_d;
  u[0] = std::min(std::max(u[0], -1.0), 1.0);
  u[1] = std::min(std::max(u[1], -1.0), 1.0);
  return u;
}

void BlueboatController::inputCallback(const usv_msgs::SpeedCourse& msg)
{
  u_d   = msg.speed;
  psi_d = msg.course;
  ROS_INFO_STREAM("Psi_d: " << psi_d << " u_d: " << u_d);
}

void BlueboatController::odomCallback(const nav_msgs::Odometry& msg)
{
  u = msg.twist.twist.linear.x;
  r = msg.twist.twist.angular.z;
}

double BlueboatController::getYaw()
{
  static tf2_ros::Buffer tfBuffer;
  static tf2_ros::TransformListener tfListener(tfBuffer);
  geometry_msgs::TransformStamped tfStamped;
  try
  {
    tfStamped = tfBuffer.lookupTransform("map", "base_link",
                                         ros::Time(0.0), ros::Duration(1.0));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    return 0.0;
  }
  return tf2::getYaw(tfStamped.transform.rotation);
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "BlueboatController");
  BlueboatController BlueboatController;
  return 0;
}