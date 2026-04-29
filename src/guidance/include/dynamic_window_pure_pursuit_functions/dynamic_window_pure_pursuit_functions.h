// ROS1 port of:
//   nav2_regulated_pure_pursuit_controller/dynamic_window_pure_pursuit_functions.hpp
//   Copyright (c) 2025 Fumiya Ohnishi, Apache License 2.0
//
// Original published under Apache-2.0:
//   https://github.com/ros-navigation/navigation2/.../dynamic_window_pure_pursuit_functions.hpp
//
// Changes from the upstream file:
//   - Replaced rclcpp/geometry_msgs::msg::Twist with ROS1 geometry_msgs/Twist.
//   - Removed the rclcpp logging include; this is a pure header-only library.
//   - Namespace renamed under blueboat_coverage to avoid colliding with any
//     downstream nav2 install on the same machine.
//
// Algorithm is unchanged. Math comments retained verbatim where useful.

#ifndef BLUEBOAT_GUIDANCE_DYNAMIC_WINDOW_PURE_PURSUIT_FUNCTIONS_H_
#define BLUEBOAT_GUIDANCE_DYNAMIC_WINDOW_PURE_PURSUIT_FUNCTIONS_H_

#include <geometry_msgs/Twist.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace blueboat_coverage
{
namespace dynamic_window_pure_pursuit
{

struct DynamicWindowBounds
{
  double max_linear_vel;
  double min_linear_vel;
  double max_angular_vel;
  double min_angular_vel;
};

inline DynamicWindowBounds computeDynamicWindow(
    const geometry_msgs::Twist& current_speed,
    const double& max_linear_vel,
    const double& min_linear_vel,
    const double& max_angular_vel,
    const double& min_angular_vel,
    const double& max_linear_accel,
    const double& max_linear_decel,
    const double& max_angular_accel,
    const double& max_angular_decel,
    const double& dt)
{
  DynamicWindowBounds dw;
  constexpr double Eps = 1e-3;

  auto compute_window =
      [&](const double& current_vel,
          const double& v_max, const double& v_min,
          const double& a_max, const double& a_dec)
      {
        double cmax = 0.0, cmin = 0.0;
        if (current_vel > Eps)
        {
          cmax = current_vel + a_max * dt;
          cmin = current_vel + a_dec * dt;
        }
        else if (current_vel < -Eps)
        {
          cmax = current_vel - a_dec * dt;
          cmin = current_vel - a_max * dt;
        }
        else
        {
          cmax = current_vel + a_max * dt;
          cmin = current_vel - a_max * dt;
        }
        return std::make_tuple(std::min(cmax, v_max), std::max(cmin, v_min));
      };

  std::tie(dw.max_linear_vel, dw.min_linear_vel) =
      compute_window(current_speed.linear.x,
                     max_linear_vel, min_linear_vel,
                     max_linear_accel, max_linear_decel);

  std::tie(dw.max_angular_vel, dw.min_angular_vel) =
      compute_window(current_speed.angular.z,
                     max_angular_vel, min_angular_vel,
                     max_angular_accel, max_angular_decel);

  return dw;
}

inline void applyRegulationToDynamicWindow(
    const double& regulated_linear_vel,
    DynamicWindowBounds& dw)
{
  double v_min = std::min(0.0, regulated_linear_vel);
  double v_max = std::max(0.0, regulated_linear_vel);

  dw.min_linear_vel = std::max(dw.min_linear_vel, v_min);
  dw.max_linear_vel = std::min(dw.max_linear_vel, v_max);

  if (dw.min_linear_vel > dw.max_linear_vel)
  {
    if (dw.min_linear_vel > v_max)
      dw.max_linear_vel = dw.min_linear_vel;
    else
      dw.min_linear_vel = dw.max_linear_vel;
  }
}

inline std::tuple<double, double> computeOptimalVelocityWithinDynamicWindow(
    const DynamicWindowBounds& dw,
    const double& curvature,
    const double& sign)
{
  double opt_v;
  double opt_w;

  // Curvature ~ 0: line angular = 0
  if (std::abs(curvature) < 1e-3)
  {
    opt_v = (sign >= 0.0) ? dw.max_linear_vel : dw.min_linear_vel;
    if (dw.min_angular_vel <= 0.0 && 0.0 <= dw.max_angular_vel)
    {
      opt_w = 0.0;
    }
    else
    {
      opt_w = (std::abs(dw.min_angular_vel) <= std::abs(dw.max_angular_vel))
                  ? dw.min_angular_vel : dw.max_angular_vel;
    }
    return std::make_tuple(opt_v, opt_w);
  }

  // Candidate intersections of the line w = curvature * v with the box
  std::pair<double, double> candidates[] = {
      {dw.min_linear_vel, curvature * dw.min_linear_vel},
      {dw.max_linear_vel, curvature * dw.max_linear_vel},
      {dw.min_angular_vel / curvature, dw.min_angular_vel},
      {dw.max_angular_vel / curvature, dw.max_angular_vel}};

  double best_v = -std::numeric_limits<double>::max() * sign;
  double best_w = 0.0;

  for (auto& cand : candidates)
  {
    double v = cand.first, w = cand.second;
    if (v >= dw.min_linear_vel && v <= dw.max_linear_vel &&
        w >= dw.min_angular_vel && w <= dw.max_angular_vel)
    {
      if (v * sign > best_v * sign)
      {
        best_v = v;
        best_w = w;
      }
    }
  }

  if (best_v != -std::numeric_limits<double>::max() * sign)
    return std::make_tuple(best_v, best_w);

  // No intersection — pick the corner closest to the line.
  const std::array<std::array<double, 2>, 4> corners = {{
      {dw.min_linear_vel, dw.min_angular_vel},
      {dw.min_linear_vel, dw.max_angular_vel},
      {dw.max_linear_vel, dw.min_angular_vel},
      {dw.max_linear_vel, dw.max_angular_vel}}};

  const double denom = std::sqrt(curvature * curvature + 1.0);
  auto dist = [&](const std::array<double, 2>& c)
  {
    return std::abs(curvature * c[0] - c[1]) / denom;
  };

  double closest = std::numeric_limits<double>::max();
  best_v = -std::numeric_limits<double>::max() * sign;
  best_w = 0.0;
  for (const auto& c : corners)
  {
    double d = dist(c);
    if (d < closest ||
        (std::abs(d - closest) <= 1e-3 && c[0] * sign > best_v * sign))
    {
      closest = d;
      best_v  = c[0];
      best_w  = c[1];
    }
  }
  return std::make_tuple(best_v, best_w);
}

inline std::tuple<double, double> computeDynamicWindowVelocities(
    const geometry_msgs::Twist& current_speed,
    const double& max_linear_vel,
    const double& min_linear_vel,
    const double& max_angular_vel,
    const double& min_angular_vel,
    const double& max_linear_accel,
    const double& max_linear_decel,
    const double& max_angular_accel,
    const double& max_angular_decel,
    const double& regulated_linear_vel,
    const double& curvature,
    const double& sign,
    const double& dt)
{
  DynamicWindowBounds dw = computeDynamicWindow(
      current_speed, max_linear_vel, min_linear_vel,
      max_angular_vel, min_angular_vel,
      max_linear_accel, max_linear_decel,
      max_angular_accel, max_angular_decel, dt);

  applyRegulationToDynamicWindow(regulated_linear_vel, dw);

  return computeOptimalVelocityWithinDynamicWindow(dw, curvature, sign);
}

}  // namespace dynamic_window_pure_pursuit
}  // namespace blueboat_coverage

#endif  // BLUEBOAT_GUIDANCE_DYNAMIC_WINDOW_PURE_PURSUIT_FUNCTIONS_H_
