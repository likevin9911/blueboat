/**
 * path_stitcher.cpp
 *
 * Glue between the search layer (A* / RRT*) and the steering layer.
 * Receives a sparse list of waypoints, sends each adjacent pair as a
 * DubinInput to the steering planner, accumulates the per-segment
 * paths, and republishes the concatenated full path.
 *
 * Subscribes:
 *   ~waypoints         (nav_msgs/Path)              from A* / RRT* / goal_to_path
 *   ~segment_path      (nav_msgs/Path)              from steering planner
 *   ~segment_curvature (std_msgs/Float32MultiArray) from HC-RS only
 *
 * Publishes:
 *   ~steering_input    (blueboat_navigation/DubinInput)  to steering planner
 *   ~full_path         (nav_msgs/Path)              to guidance
 *   ~full_curvatures   (std_msgs/Float32MultiArray) to guidance (hc_rs only)
 *
 * Behavior rules:
 *  - Receiving a new ~waypoints message ALWAYS supersedes any in-flight
 *    work. Segments collected for an old request are discarded.
 *  - The stitcher NEVER publishes an empty path. If all segments time
 *    out or no segments are collected, guidance keeps the last good
 *    path it had — better than zeroing it out.
 *  - Sequential processing: segment k must complete before k+1 is sent.
 *
 * Params (private):
 *   ~segment_timeout   [s]    default 1.0
 *   ~expect_curvatures [bool] default false. Set true only for hc_rs.
 */

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32MultiArray.h>

#include <blueboat_navigation/DubinInput.h>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace blueboat_navigation
{

class PathStitcher
{
public:
  PathStitcher()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhP("~");

    m_segmentTimeout   = nhP.param("segment_timeout", 1.0);
    m_expectCurvatures = nhP.param("expect_curvatures", false);

    m_waypointsSub = nhP.subscribe("waypoints", 1,
                                   &PathStitcher::onWaypoints, this);
    m_segmentSub   = nhP.subscribe("segment_path", 10,
                                   &PathStitcher::onSegment, this);

    if (m_expectCurvatures)
    {
      m_curvSub = nhP.subscribe("segment_curvature", 10,
                                &PathStitcher::onCurvature, this);
      m_curvPub = nhP.advertise<std_msgs::Float32MultiArray>(
          "full_curvatures", 1, true);
    }

    m_inputPub = nhP.advertise<blueboat_navigation::DubinInput>(
        "steering_input", 10);
    m_pathPub  = nhP.advertise<nav_msgs::Path>("full_path", 1, true);

    m_timer = nh.createTimer(ros::Duration(0.05),
                             &PathStitcher::tick, this);

    ROS_INFO("[path_stitcher] timeout=%.2fs  expect_curvatures=%s",
             m_segmentTimeout, m_expectCurvatures ? "true" : "false");
  }

private:
  // -------- callbacks --------
  void onWaypoints(const nav_msgs::Path& path)
  {
    std::lock_guard<std::mutex> lk(m_mu);

    if (path.poses.size() < 2)
    {
      ROS_WARN("[path_stitcher] received %zu waypoints; need at least 2",
               path.poses.size());
      return;
    }

    // Bump sequence number — any in-flight segment response from a prior
    // request will be ignored when it arrives.
    ++m_jobSeq;

    m_segments.clear();
    m_curvatures.clear();
    m_pendingPairs.clear();
    m_pendingCurv.clear();
    m_frameId = path.header.frame_id;
    m_awaitingResp       = false;
    m_haveSegForCurrent  = false;
    m_haveCurvForCurrent = false;

    for (size_t i = 0; i + 1 < path.poses.size(); ++i)
    {
      m_pendingPairs.push_back({path.poses[i], path.poses[i + 1]});
    }

    ROS_INFO("[path_stitcher] queued %zu segments (job %u)",
             m_pendingPairs.size(), m_jobSeq);
  }

  void onSegment(const nav_msgs::Path& seg)
  {
    std::lock_guard<std::mutex> lk(m_mu);
    if (!m_awaitingResp) return;

    m_segments.push_back(seg);

    if (!m_expectCurvatures)
    {
      finishCurrentSegment();
      return;
    }

    if (m_haveCurvForCurrent)
    {
      finishCurrentSegment();
    }
    else
    {
      m_haveSegForCurrent = true;
    }
  }

  void onCurvature(const std_msgs::Float32MultiArray& msg)
  {
    std::lock_guard<std::mutex> lk(m_mu);
    if (!m_awaitingResp || !m_expectCurvatures) return;

    m_pendingCurv = msg.data;
    if (m_haveSegForCurrent)
    {
      finishCurrentSegment();
    }
    else
    {
      m_haveCurvForCurrent = true;
    }
  }

  void finishCurrentSegment()
  {
    if (m_expectCurvatures)
    {
      m_curvatures.insert(m_curvatures.end(),
                          m_pendingCurv.begin(), m_pendingCurv.end());
      m_pendingCurv.clear();
    }
    m_awaitingResp        = false;
    m_haveSegForCurrent   = false;
    m_haveCurvForCurrent  = false;
    sendNextOrFinalize();
  }

  void tick(const ros::TimerEvent&)
  {
    std::lock_guard<std::mutex> lk(m_mu);
    if (!m_awaitingResp && !m_pendingPairs.empty())
    {
      sendNextOrFinalize();
      return;
    }

    if (m_awaitingResp &&
        (ros::Time::now() - m_segmentSentAt).toSec() > m_segmentTimeout)
    {
      ROS_WARN("[path_stitcher] segment %zu timed out (job %u)",
               m_segments.size(), m_jobSeq);
      m_awaitingResp        = false;
      m_haveSegForCurrent   = false;
      m_haveCurvForCurrent  = false;
      if (!m_pendingPairs.empty()) m_pendingPairs.pop_front();
      sendNextOrFinalize();
    }
  }

  void sendNextOrFinalize()
  {
    if (m_pendingPairs.empty())
    {
      publishConcatenated();
      return;
    }

    auto pair = m_pendingPairs.front();
    m_pendingPairs.pop_front();

    blueboat_navigation::DubinInput msg;
    msg.header.stamp    = ros::Time::now();
    msg.header.frame_id = m_frameId.empty() ? std::string("map") : m_frameId;
    msg.start = pair.first;
    msg.end   = pair.second;
    m_inputPub.publish(msg);

    m_awaitingResp  = true;
    m_segmentSentAt = ros::Time::now();
  }

  void publishConcatenated()
  {
    // Critical: never publish an empty path. Doing so wipes whatever
    // guidance was following.
    if (m_segments.empty())
    {
      ROS_WARN("[path_stitcher] no segments collected for job %u; "
               "leaving previous path in place",
               m_jobSeq);
      return;
    }

    nav_msgs::Path full;
    full.header.stamp    = ros::Time::now();
    full.header.frame_id = m_frameId.empty() ? std::string("map") : m_frameId;

    size_t total = 0;
    for (const auto& s : m_segments) total += s.poses.size();
    full.poses.reserve(total);

    for (size_t s = 0; s < m_segments.size(); ++s)
    {
      // Skip the first pose of subsequent segments — it duplicates the
      // last pose of the previous segment (the join point).
      size_t startI = (s == 0) ? 0 : 1;
      for (size_t i = startI; i < m_segments[s].poses.size(); ++i)
      {
        full.poses.push_back(m_segments[s].poses[i]);
      }
    }

    if (full.poses.size() < 2)
    {
      ROS_WARN("[path_stitcher] concatenated path has %zu poses; not publishing",
               full.poses.size());
      return;
    }

    m_pathPub.publish(full);
    ROS_INFO("[path_stitcher] published full path: %zu poses across %zu segments (job %u)",
             full.poses.size(), m_segments.size(), m_jobSeq);

    if (m_expectCurvatures)
    {
      std_msgs::Float32MultiArray curvOut;
      curvOut.data = m_curvatures;
      m_curvPub.publish(curvOut);
    }
  }

  // -------- state --------
  ros::Subscriber m_waypointsSub;
  ros::Subscriber m_segmentSub;
  ros::Subscriber m_curvSub;
  ros::Publisher  m_inputPub;
  ros::Publisher  m_pathPub;
  ros::Publisher  m_curvPub;
  ros::Timer      m_timer;

  std::mutex m_mu;

  std::deque<std::pair<geometry_msgs::PoseStamped,
                       geometry_msgs::PoseStamped>> m_pendingPairs;
  std::vector<nav_msgs::Path> m_segments;
  std::vector<float>          m_curvatures;
  std::vector<float>          m_pendingCurv;

  std::string m_frameId;
  bool      m_awaitingResp{false};
  bool      m_haveSegForCurrent{false};
  bool      m_haveCurvForCurrent{false};
  ros::Time m_segmentSentAt;
  unsigned  m_jobSeq{0};

  double m_segmentTimeout;
  bool   m_expectCurvatures;
};

} // namespace blueboat_navigation

int main(int argc, char** argv)
{
  ros::init(argc, argv, "path_stitcher");
  blueboat_navigation::PathStitcher node;
  ros::spin();
  return 0;
}