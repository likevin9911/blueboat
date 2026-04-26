#pragma once

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <vector>
#include <array>
#include <adaptive_clustering/ClusterArray.h>

namespace buoy_course
{

// ------------------------------------------------------------
// Single tracked buoy — simple Kalman filter on (x, y) position
// ------------------------------------------------------------
struct TrackedBuoy
{
  int    id;               // assigned ID (1-based, -1 = unconfirmed)
  double x, y;            // filtered position in map frame
  double vx, vy;          // velocity estimate

  // Kalman state: [x, y, vx, vy]
  std::array<double, 4>  state;
  // 4x4 covariance (row-major)
  std::array<double, 16> P;

  int    hit_count;        // number of times detection matched this track
  int    miss_count;       // consecutive frames with no matching detection
  bool   confirmed;        // true once hit_count >= confirmation_threshold
  bool   active;

  ros::Time last_seen;

  TrackedBuoy();
  void predict(double dt);
  void update(double meas_x, double meas_y, double meas_noise);
};

// ------------------------------------------------------------
// BuoyTracker node
// ------------------------------------------------------------
class BuoyTracker
{
public:
  BuoyTracker();
  ~BuoyTracker() = default;

private:
  // ROS I/O
  ros::NodeHandle m_nh;
  ros::NodeHandle m_nhP;
  ros::Subscriber m_clusterSub;
  ros::Publisher  m_buoyPub;       // geometry_msgs/PoseArray  — confirmed buoys in map frame
  ros::Publisher  m_markerPub;     // visualization_msgs/MarkerArray — rviz

  // TF
  tf2_ros::Buffer            m_tfBuffer;
  tf2_ros::TransformListener m_tfListener;

  // Parameters
  double m_assocRadius;         // max distance to associate a detection to a track (m)
  double m_measNoise;           // measurement noise std dev (m)
  int    m_confirmThresh;       // hit_count needed to confirm a buoy
  int    m_maxMissCount;        // miss_count before track is dropped
  int    m_maxBuoys;            // expected number of buoys on the course
  double m_maxRange;            // ignore detections beyond this range (m)
  double m_minRange;            // ignore detections closer than this (boat itself)
  std::string m_mapFrame;

  // State
  std::vector<TrackedBuoy> m_tracks;
  int m_nextTrackId;
  ros::Time m_lastUpdateTime;

  // Callbacks
  void clusterCb(const adaptive_clustering::ClusterArray::ConstPtr& msg);

  // Helpers
  void   predictAll(double dt);
  int    findBestTrack(double x, double y);
  void   createNewTrack(double x, double y);
  void   removeDeadTracks();
  void   publishBuoys();
  void   publishMarkers();
};

} // namespace buoy_course
