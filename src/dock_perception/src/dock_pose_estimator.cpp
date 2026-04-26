// dock_pose_estimator_node.cpp
// Subscribes to adaptive_clustering/ClusterArray
// Runs L-shape fitting on each cluster
// Filters by size to isolate the dock
// Publishes geometry_msgs/PoseStamped on /dock_pose

#include <ros/ros.h>

// Messages
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Header.h>

// Adaptive clustering message
#include <adaptive_clustering/ClusterArray.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>

// TF2
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

// Eigen
#include <Eigen/Core>

// Our clean L-shape estimator — no obsdet, no autoware
#include "dock_perception/lshape_estimator.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <map>
#include <functional>


// -----------------------------------------------------------------------
// Merge clusters whose 2-D centroids are within merge_radius of each other.
// -----------------------------------------------------------------------
static std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
mergeClusters(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds,
              double merge_radius)
{
  int n = (int)clouds.size();
  if (n == 0) return {};

  std::vector<double> cx(n), cy(n);
  for (int i = 0; i < n; ++i)
  {
    double sx = 0, sy = 0;
    for (const auto& p : *clouds[i]) { sx += p.x; sy += p.y; }
    cx[i] = sx / clouds[i]->size();
    cy[i] = sy / clouds[i]->size();
  }

  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int x) -> int {
    return parent[x] == x ? x : parent[x] = find(parent[x]);
  };

  double r2 = merge_radius * merge_radius;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
    {
      double dx = cx[i] - cx[j], dy = cy[i] - cy[j];
      if (dx*dx + dy*dy <= r2)
      {
        int pi = find(i), pj = find(j);
        if (pi != pj) parent[pi] = pj;
      }
    }

  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> groups;
  for (int i = 0; i < n; ++i)
  {
    int root = find(i);
    if (groups.find(root) == groups.end())
      groups[root] = pcl::PointCloud<pcl::PointXYZ>::Ptr(
                       new pcl::PointCloud<pcl::PointXYZ>);
    *groups[root] += *clouds[i];
  }

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> result;
  for (auto& kv : groups) result.push_back(kv.second);
  return result;
}

class DockPoseEstimator
{
public:
  DockPoseEstimator() : nh_(), pnh_("~"), tf2_listener_(tf2_buffer_)
  {
    // --- Parameters ---
    pnh_.param<std::string>("target_frame",    target_frame_,    "map");
    pnh_.param<std::string>("criterion",       criterion_,       "VARIANCE");
    pnh_.param<double>("min_dock_length",      min_dock_length_, 1.0);
    pnh_.param<double>("max_dock_length",      max_dock_length_, 20.0);
    pnh_.param<double>("min_dock_width",       min_dock_width_,  0.3);
    pnh_.param<double>("max_dock_width",       max_dock_width_,  6.0);
    pnh_.param<double>("min_cluster_points",   min_points_,      10.0);
    pnh_.param<double>("merge_radius",         merge_radius_,    3.0);
    pnh_.param<int>("lock_after_frames",       lock_after_frames_, 10);

    // --- Subscribers ---
    cluster_sub_ = nh_.subscribe(
      "adaptive_clustering/clusters", 1,
      &DockPoseEstimator::clusterCb, this);

    // --- Publishers ---
    dock_pub_   = nh_.advertise<geometry_msgs::PoseStamped>("dock_pose", 10);
    dims_pub_   = nh_.advertise<geometry_msgs::Vector3Stamped>("dock_dims", 10);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("dock_markers", 10);

    ROS_INFO("[DockPoseEstimator] Ready.");
    ROS_INFO("[DockPoseEstimator] target_frame : %s", target_frame_.c_str());
    ROS_INFO("[DockPoseEstimator] criterion    : %s", criterion_.c_str());
    ROS_INFO("[DockPoseEstimator] dock length  : %.1f - %.1f m", min_dock_length_, max_dock_length_);
    ROS_INFO("[DockPoseEstimator] dock width   : %.1f - %.1f m", min_dock_width_,  max_dock_width_);
    ROS_INFO("[DockPoseEstimator] merge_radius : %.1f m  lock_after: %d frames", merge_radius_, lock_after_frames_);
  }

private:

  // -----------------------------------------------------------------------
  // Struct to hold fitted result for one cluster
  // -----------------------------------------------------------------------
  struct FittedCluster
  {
    double pos_x, pos_y, pos_z;
    double yaw;
    double length, width;
    double score;   // length * width — used to pick best candidate
  };

  // -----------------------------------------------------------------------
  // Calculate bounding box center, dimensions and yaw from theta_optim
  // Ported directly from LShapeFittingNode::calculateDimPos
  // but returning plain doubles instead of obsdet_msgs
  // -----------------------------------------------------------------------
  bool calcDimPos(const pcl::PointCloud<pcl::PointXYZ>& cluster,
                  double theta_star,
                  FittedCluster& out)
  {
    constexpr double ep = 0.001;

    // Centroid z
    double sum_z = 0.0;
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::lowest();
    for (const auto& pt : cluster)
    {
      sum_z += pt.z;
      min_z = std::min(min_z, (double)pt.z);
      max_z = std::max(max_z, (double)pt.z);
    }
    out.pos_z = sum_z / cluster.size();

    // Project points onto fitted axes
    Eigen::Vector2d e1, e2;
    e1 <<  std::cos(theta_star), std::sin(theta_star);
    e2 << -std::sin(theta_star), std::cos(theta_star);

    std::vector<double> C1, C2;
    for (const auto& pt : cluster)
    {
      C1.push_back(pt.x * e1.x() + pt.y * e1.y());
      C2.push_back(pt.x * e2.x() + pt.y * e2.y());
    }

    const double c1_min = *std::min_element(C1.begin(), C1.end());
    const double c1_max = *std::max_element(C1.begin(), C1.end());
    const double c2_min = *std::min_element(C2.begin(), C2.end());
    const double c2_max = *std::max_element(C2.begin(), C2.end());

    // Rectangle edges: a*x + b*y = c
    const double a1 =  std::cos(theta_star), b1 = std::sin(theta_star),  c1v = c1_min;
    const double a2 = -std::sin(theta_star), b2 = std::cos(theta_star),  c2v = c2_min;
    const double a3 =  std::cos(theta_star), b3 = std::sin(theta_star),  c3v = c1_max;
    const double a4 = -std::sin(theta_star), b4 = std::cos(theta_star),  c4v = c2_max;

    // Two opposite corners
    double denom1 = (a2 * b1 - a1 * b2);
    double denom2 = (a4 * b3 - a3 * b4);
    if (std::fabs(denom1) < ep || std::fabs(denom2) < ep) return false;

    double ix1 = (b1 * c2v - b2 * c1v) / denom1;
    double iy1 = (a1 * c2v - a2 * c1v) / (a1 * b2 - a2 * b1);
    double ix2 = (b3 * c4v - b4 * c3v) / denom2;
    double iy2 = (a3 * c4v - a4 * c3v) / (a3 * b4 - a4 * b3);

    // Dimensions
    Eigen::Vector2d ex(a1 / std::hypot(a1, b1), b1 / std::hypot(a1, b1));
    Eigen::Vector2d ey(a2 / std::hypot(a2, b2), b2 / std::hypot(a2, b2));
    Eigen::Vector2d diag(ix1 - ix2, iy1 - iy2);

    double dim_x = std::fabs(ex.dot(diag));
    double dim_y = std::fabs(ey.dot(diag));

    out.length = std::max(dim_x, dim_y);
    out.width  = std::min(dim_x, dim_y);

    // Center
    out.pos_x = (ix1 + ix2) / 2.0;
    out.pos_y = (iy1 + iy2) / 2.0;

    // Yaw — along the long axis
    if (dim_x >= dim_y)
      out.yaw = std::atan2(e1.y(), e1.x());
    else
      out.yaw = std::atan2(e1.y(), e1.x()) + M_PI_2;

    out.score = out.length * out.width;
    return true;
  }

  // -----------------------------------------------------------------------
  // Main callback
  // -----------------------------------------------------------------------
  void clusterCb(const adaptive_clustering::ClusterArray::ConstPtr& msg)
  {
    if (msg->clusters.empty()) return;

    // If already locked, just re-publish the stored pose
    if (dock_locked_)
    {
      locked_pose_.header.stamp = msg->header.stamp;
      dock_pub_.publish(locked_pose_);
      // Publish locked dimensions
      geometry_msgs::Vector3Stamped dims_msg;
      dims_msg.header = locked_pose_.header;
      dims_msg.vector.x = locked_length_;
      dims_msg.vector.y = locked_width_;
      dims_pub_.publish(dims_msg);
      publishMarkers(locked_pose_, locked_length_, locked_width_);
      return;
    }

    // Convert + merge incoming clusters
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> raw_clouds;
    for (const auto& cloud_msg : msg->clusters)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::fromROSMsg(cloud_msg, *c);
      if (!c->empty()) raw_clouds.push_back(c);
    }
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> merged =
      mergeClusters(raw_clouds, merge_radius_);

    orientation_calc fitter(criterion_);
    std::vector<FittedCluster> candidates;

    for (const auto& cluster : merged)
    {
      if ((int)cluster->size() < (int)min_points_) continue;

      // L-shape fit
      double theta_optim = 0.0;
      if (!fitter.LshapeFitting(*cluster, theta_optim)) continue;

      // Get bounding box
      FittedCluster fc;
      if (!calcDimPos(*cluster, theta_optim, fc)) continue;

      // Size filter — reject anything that can't be the dock
      if (fc.length < min_dock_length_ || fc.length > max_dock_length_) continue;
      if (fc.width  < min_dock_width_  || fc.width  > max_dock_width_)  continue;

      candidates.push_back(fc);
    }

    if (candidates.empty())
    {
      ROS_WARN_THROTTLE(5.0, "[DockPoseEstimator] No dock candidate found.");
      return;
    }

    // Pick largest bounding box area — most likely the dock
    const FittedCluster& best = *std::max_element(
      candidates.begin(), candidates.end(),
      [](const FittedCluster& a, const FittedCluster& b)
      { return a.score < b.score; });

    // Build PoseStamped in sensor frame
    geometry_msgs::PoseStamped pose_sensor;
    pose_sensor.header = msg->header;
    pose_sensor.pose.position.x = best.pos_x;
    pose_sensor.pose.position.y = best.pos_y;
    pose_sensor.pose.position.z = best.width / 2.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, best.yaw);
    pose_sensor.pose.orientation = tf2::toMsg(q);

    // Transform to target frame (map/odom)
    geometry_msgs::PoseStamped pose_out = pose_sensor;
    if (pose_sensor.header.frame_id != target_frame_)
    {
      try
      {
        tf2_buffer_.transform(pose_sensor, pose_out, target_frame_,
                              ros::Duration(0.1));
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN_THROTTLE(1.0,
          "[DockPoseEstimator] TF failed (%s->%s): %s",
          pose_sensor.header.frame_id.c_str(),
          target_frame_.c_str(), ex.what());
        return;
      }
    }

    dock_pub_.publish(pose_out);

    // Publish bounding box dimensions (length x width)
    {
      geometry_msgs::Vector3Stamped dims_msg;
      dims_msg.header = pose_out.header;
      dims_msg.vector.x = best.length;
      dims_msg.vector.y = best.width;
      dims_pub_.publish(dims_msg);
    }

    // Accumulate consistent detections then lock
    consistent_frames_++;
    ROS_INFO_THROTTLE(1.0, "[DockPoseEstimator] consistent=%d/%d  pos=(%.2f,%.2f)  %.1fx%.1fm",
                      consistent_frames_, lock_after_frames_,
                      pose_out.pose.position.x, pose_out.pose.position.y,
                      best.length, best.width);
    if (consistent_frames_ >= lock_after_frames_)
    {
      dock_locked_   = true;
      locked_pose_   = pose_out;
      locked_length_ = best.length;
      locked_width_  = best.width;
      ROS_INFO("[DockPoseEstimator] *** DOCK LOCKED at (%.2f, %.2f) yaw=%.1f deg ***",
               pose_out.pose.position.x, pose_out.pose.position.y,
               best.yaw * 180.0 / M_PI);
    }

    ROS_DEBUG("[DockPoseEstimator] Dock at (%.2f, %.2f) yaw=%.2f deg  "
              "size=%.2fx%.2f m",
              pose_out.pose.position.x,
              pose_out.pose.position.y,
              best.yaw * 180.0 / M_PI,
              best.length, best.width);

    publishMarkers(pose_out, best.length, best.width);
  }

  // -----------------------------------------------------------------------
  // RViz markers — bounding box cube + heading arrow
  // -----------------------------------------------------------------------
  void publishMarkers(const geometry_msgs::PoseStamped& pose,
                      double length, double width)
  {
    visualization_msgs::MarkerArray arr;

    // Green semi-transparent box
    visualization_msgs::Marker box;
    box.header    = pose.header;
    box.ns        = "dock_box";
    box.id        = 0;
    box.type      = visualization_msgs::Marker::CUBE;
    box.action    = visualization_msgs::Marker::ADD;
    box.pose      = pose.pose;
    box.scale.x   = length;
    box.scale.y   = width;
    box.scale.z   = 0.5;
    box.color.r   = 0.0f;
    box.color.g   = 0.8f;
    box.color.b   = 0.2f;
    box.color.a   = 0.4f;
    box.lifetime  = ros::Duration(0.3);
    arr.markers.push_back(box);

    // Orange arrow showing dock heading direction
    visualization_msgs::Marker arrow;
    arrow.header   = pose.header;
    arrow.ns       = "dock_heading";
    arrow.id       = 1;
    arrow.type     = visualization_msgs::Marker::ARROW;
    arrow.action   = visualization_msgs::Marker::ADD;
    arrow.pose     = pose.pose;
    arrow.scale.x  = length * 0.7;  // shaft length
    arrow.scale.y  = 0.25;
    arrow.scale.z  = 0.25;
    arrow.color.r  = 1.0f;
    arrow.color.g  = 0.4f;
    arrow.color.b  = 0.0f;
    arrow.color.a  = 0.9f;
    arrow.lifetime = ros::Duration(0.3);
    arr.markers.push_back(arrow);

    // Text label with dimensions
    visualization_msgs::Marker text;
    text.header          = pose.header;
    text.ns              = "dock_label";
    text.id              = 2;
    text.type            = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text.action          = visualization_msgs::Marker::ADD;
    text.pose            = pose.pose;
    text.pose.position.z += 1.0;
    text.scale.z         = 0.4;
    text.color.r = text.color.g = text.color.b = 1.0f;
    text.color.a         = 1.0f;
    char buf[64];
    snprintf(buf, sizeof(buf), "dock %.1fx%.1fm", length, width);
    text.text            = std::string(buf);
    text.lifetime        = ros::Duration(0.3);
    arr.markers.push_back(text);

    marker_pub_.publish(arr);
  }

  // -----------------------------------------------------------------------
  // Members
  // -----------------------------------------------------------------------
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber cluster_sub_;
  ros::Publisher  dock_pub_;
  ros::Publisher  dims_pub_;
  ros::Publisher  marker_pub_;

  tf2_ros::Buffer            tf2_buffer_;
  tf2_ros::TransformListener tf2_listener_;

  std::string target_frame_;
  std::string criterion_;
  double min_dock_length_, max_dock_length_;
  double min_dock_width_,  max_dock_width_;
  double min_points_;
  double merge_radius_;

  // Latch state
  int  lock_after_frames_;
  int  consistent_frames_  = 0;
  bool dock_locked_        = false;
  geometry_msgs::PoseStamped locked_pose_;
  double locked_length_ = 0.0, locked_width_ = 0.0;
};

// -----------------------------------------------------------------------
int main(int argc, char** argv)
{
  ros::init(argc, argv, "dock_pose_estimator");
  DockPoseEstimator node;
  ros::spin();
  return 0;
}