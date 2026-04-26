#pragma once

#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <vector>
#include <fstream>

namespace buoy_course
{

struct BuoyMetric
{
  int    buoy_id;
  double min_clearance;    // closest approach during rounding (m)
  double max_clearance;    // furthest point during rounding (m)
  double sum_clearance;
  int    sample_count;
  double avg_clearance() const
  {
    return sample_count > 0 ? sum_clearance / sample_count : 0.0;
  }
};

class MetricsNode
{
public:
  MetricsNode();
  ~MetricsNode();

private:
  ros::NodeHandle m_nh;
  ros::NodeHandle m_nhP;

  ros::Subscriber m_buoySub;       // confirmed buoy positions
  ros::Subscriber m_activeIdSub;   // which buoy is currently being rounded
  ros::Publisher  m_statusPub;     // std_msgs/String — live status for rviz/rqt

  tf2_ros::Buffer            m_tfBuffer;
  tf2_ros::TransformListener m_tfListener;

  // Parameters
  std::string m_mapFrame;
  double      m_roundingRadius;   // distance at which we consider "rounding" active
  std::string m_logFile;

  // State
  std::vector<geometry_msgs::Pose> m_buoys;  // latest confirmed buoy positions
  int    m_activeBuoyId;    // 1-based, -1 = none
  bool   m_lapStarted;
  bool   m_lapFinished;
  ros::Time m_lapStartTime;
  ros::Time m_lapEndTime;
  double    m_lapTime;

  std::vector<BuoyMetric> m_metrics;    // one entry per confirmed buoy

  // Start-line crossing detection
  double m_startX, m_startY;
  double m_startLineHalfWidth;          // half-width of virtual start gate
  bool   m_crossedStartOutbound;
  geometry_msgs::Pose m_lastPose;
  bool   m_hasPose;

  ros::Timer m_updateTimer;

  void buoyCb(const geometry_msgs::PoseArray::ConstPtr& msg);
  void activeIdCb(const std_msgs::Int32::ConstPtr& msg);
  void timerCb(const ros::TimerEvent&);

  void updateClearance(double rx, double ry);
  bool checkStartLineCrossing(double rx, double ry);
  void printReport();
  void saveReport();
};

} // namespace buoy_course
