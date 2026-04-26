/**
 * vfh_node.cpp
 *
 * Local reactive obstacle avoidance using a Vector Field Histogram (VFH).
 * Sits between guidance and the controller. Reads the desired
 * speed/heading from guidance and the latest laser scan, builds a polar
 * obstacle histogram around the boat, finds the candidate heading
 * closest to the desired heading that's not blocked, and republishes
 * (possibly modified) speed/heading to the controller.
 *
 * If the desired heading is clear, the message passes through unchanged.
 *
 * Subscribes:
 *   ~speed_heading_in  (usv_msgs/SpeedCourse)   from guidance
 *   ~scan              (sensor_msgs/LaserScan)  from the laser
 *
 * Publishes:
 *   ~speed_heading_out (usv_msgs/SpeedCourse)   to the controller
 *
 * Params (private):
 *   ~window_radius   [m]    radius of the local histogram window. default 6.0
 *   ~bin_resolution  [deg]  angular resolution per histogram bin. default 5.0
 *   ~obstacle_distance_threshold [m]
 *                            range below which a return marks a bin as
 *                            blocked. default = window_radius.
 *   ~hysteresis      [bool] keep the previous chosen direction if it's
 *                            still clear, even if a slightly closer-to-
 *                            desired bin opens up, to avoid chattering.
 *                            default true.
 *   ~speed_scale_when_avoiding [0..1]
 *                            multiplier applied to commanded speed when
 *                            VFH is overriding. default 0.5.
 *   ~global_frame    [str]  default "map"
 *   ~robot_frame     [str]  default "base_link"
 *
 * Note: VFH operates in the body frame (heading is relative to the
 * boat's bow). The desired heading from guidance is in the global frame,
 * so we rotate it into the body frame for the lookup, then back to
 * global for the output.
 */

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <usv_msgs/SpeedCourse.h>

#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace blueboat_coverage
{

class VFHNode
{
public:
  VFHNode()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    m_windowRadius          = nhP.param("window_radius", 6.0);
    double binDeg           = nhP.param("bin_resolution", 5.0);
    m_binResolution         = binDeg * M_PI / 180.0;
    m_obstacleDistThresh    = nhP.param("obstacle_distance_threshold",
                                        m_windowRadius);
    m_hysteresis            = nhP.param("hysteresis", true);
    m_speedScaleWhenAvoid   = nhP.param("speed_scale_when_avoiding", 0.5);
    nhP.param<std::string>("global_frame", m_globalFrame, "map");
    nhP.param<std::string>("robot_frame",  m_robotFrame,  "base_link");

    m_numBins = static_cast<int>(std::ceil(2.0 * M_PI / m_binResolution));
    m_histogram.assign(m_numBins, 0);

    m_inSub   = nhP.subscribe("speed_heading_in", 10,
                              &VFHNode::onSpeedHeading, this);
    m_scanSub = nhP.subscribe("scan", 10,
                              &VFHNode::onScan, this);

    m_outPub  = nhP.advertise<usv_msgs::SpeedCourse>("speed_heading_out", 10);

    m_tfListener = std::make_shared<tf2_ros::TransformListener>(m_tfBuffer);

    ROS_INFO("[vfh_node] window=%.1fm  bins=%d (%.1f deg)  obstacle_thresh=%.1fm",
             m_windowRadius, m_numBins, binDeg, m_obstacleDistThresh);
  }

private:
  // -------- callbacks --------
  void onScan(const sensor_msgs::LaserScan& scan)
  {
    // Rebuild histogram in body frame: bin i covers angles
    // [i*binRes - π, (i+1)*binRes - π).
    std::vector<int> hist(m_numBins, 0);

    double a = scan.angle_min;
    for (size_t k = 0; k < scan.ranges.size(); ++k, a += scan.angle_increment)
    {
      double r = scan.ranges[k];
      if (!std::isfinite(r)) continue;
      if (r < scan.range_min || r > scan.range_max) continue;
      if (r > m_obstacleDistThresh) continue;
      if (r > m_windowRadius) continue;

      // Wrap angle to [-π, π]
      double aw = a;
      while (aw >  M_PI) aw -= 2 * M_PI;
      while (aw < -M_PI) aw += 2 * M_PI;

      int bin = static_cast<int>(std::floor((aw + M_PI) / m_binResolution));
      bin = std::max(0, std::min(m_numBins - 1, bin));
      hist[bin]++;
    }

    m_histogram = std::move(hist);
    m_haveScan = true;
  }

  void onSpeedHeading(const usv_msgs::SpeedCourse& in)
  {
    if (!m_haveScan)
    {
      // No scan yet: pass through.
      m_outPub.publish(in);
      return;
    }

    // Get current robot yaw to convert global → body frame.
    double psi;
    if (!getRobotYaw(psi))
    {
      ROS_WARN_THROTTLE(2.0,
          "[vfh_node] no tf yet; passing through");
      m_outPub.publish(in);
      return;
    }

    double desiredGlobal = in.course;
    double desiredBody   = desiredGlobal - psi;
    while (desiredBody >  M_PI) desiredBody -= 2 * M_PI;
    while (desiredBody < -M_PI) desiredBody += 2 * M_PI;

    int desiredBin = angleToBin(desiredBody);
    bool desiredClear = (m_histogram[desiredBin] == 0);

    // If clear, pass through.
    if (desiredClear)
    {
      m_outPub.publish(in);
      m_lastChoseBin = -1;
      return;
    }

    // Find nearest unblocked bin to the desired one.
    int chosen = -1;
    int bestDist = std::numeric_limits<int>::max();
    for (int i = 0; i < m_numBins; ++i)
    {
      if (m_histogram[i] != 0) continue;
      int d = circularBinDist(i, desiredBin);
      if (d < bestDist)
      {
        bestDist = d;
        chosen   = i;
      }
    }

    if (chosen < 0)
    {
      // Boxed in: command zero speed, hold current heading.
      ROS_WARN_THROTTLE(1.0,
          "[vfh_node] no clear direction; commanding stop");
      usv_msgs::SpeedCourse out = in;
      out.speed  = 0.0;
      out.course = psi;
      m_outPub.publish(out);
      return;
    }

    // Hysteresis: if last chosen bin is still clear and within a few
    // bins of the new candidate, keep it to suppress chatter.
    if (m_hysteresis && m_lastChoseBin >= 0 &&
        m_histogram[m_lastChoseBin] == 0 &&
        circularBinDist(m_lastChoseBin, desiredBin) <= bestDist + 1)
    {
      chosen = m_lastChoseBin;
    }
    m_lastChoseBin = chosen;

    double chosenBody   = binToAngle(chosen);
    double chosenGlobal = chosenBody + psi;
    while (chosenGlobal >  M_PI) chosenGlobal -= 2 * M_PI;
    while (chosenGlobal < -M_PI) chosenGlobal += 2 * M_PI;

    usv_msgs::SpeedCourse out = in;
    out.course = chosenGlobal;
    out.speed  = in.speed * m_speedScaleWhenAvoid;
    m_outPub.publish(out);

    ROS_INFO_THROTTLE(0.5,
        "[vfh_node] override: desired=%.1f deg -> %.1f deg (body)",
        desiredBody * 180.0 / M_PI, chosenBody * 180.0 / M_PI);
  }

  // -------- helpers --------
  int angleToBin(double a) const
  {
    while (a >  M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    int bin = static_cast<int>(std::floor((a + M_PI) / m_binResolution));
    return std::max(0, std::min(m_numBins - 1, bin));
  }

  double binToAngle(int bin) const
  {
    return -M_PI + (bin + 0.5) * m_binResolution;
  }

  int circularBinDist(int a, int b) const
  {
    int d = std::abs(a - b);
    return std::min(d, m_numBins - d);
  }

  bool getRobotYaw(double& yaw)
  {
    geometry_msgs::TransformStamped tf;
    try
    {
      tf = m_tfBuffer.lookupTransform(m_globalFrame, m_robotFrame,
                                       ros::Time(0), ros::Duration(0.0));
    }
    catch (tf2::TransformException&)
    {
      return false;
    }
    yaw = tf2::getYaw(tf.transform.rotation);
    return true;
  }

  // -------- members --------
  ros::Subscriber m_inSub;
  ros::Subscriber m_scanSub;
  ros::Publisher  m_outPub;

  tf2_ros::Buffer m_tfBuffer;
  std::shared_ptr<tf2_ros::TransformListener> m_tfListener;

  std::vector<int> m_histogram;
  int  m_numBins;
  bool m_haveScan{false};
  int  m_lastChoseBin{-1};

  double m_windowRadius;
  double m_binResolution;
  double m_obstacleDistThresh;
  bool   m_hysteresis;
  double m_speedScaleWhenAvoid;
  std::string m_globalFrame;
  std::string m_robotFrame;
};

} // namespace blueboat_coverage

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vfh_node");
  blueboat_coverage::VFHNode node;
  ros::spin();
  return 0;
}
