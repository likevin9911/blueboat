// docking_metric_node.cpp
//
// Measures docking accuracy.
// - Subscribes to /dock_pose (desired docking reference from estimator)
// - Subscribes to /docking/docked (Bool, true when docked)
// - On first dock_pose: records the desired goal pose
// - On docked=true: records the robot's actual pose via TF
// - Publishes the position error (m) and heading error (deg) to /docking/metric
// - Also logs to ROS_INFO

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>

#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

#include <cmath>

class DockingMetric
{
public:
  DockingMetric() : nh_(), pnh_("~"), tf_listener_(tf_buffer_)
  {
    pnh_.param<std::string>("map_frame", map_frame_, "map");

    dock_sub_   = nh_.subscribe("/dock_pose", 5, &DockingMetric::dockCb, this);
    docked_sub_ = nh_.subscribe("/docking/docked", 5, &DockingMetric::dockedCb, this);
    dims_sub_   = nh_.subscribe("/dock_dims", 5, &DockingMetric::dimsCb, this);

    pos_err_pub_ = nh_.advertise<std_msgs::Float64>("/docking/metric/position_error", 5, true);
    yaw_err_pub_ = nh_.advertise<std_msgs::Float64>("/docking/metric/heading_error", 5, true);
    time_pub_    = nh_.advertise<std_msgs::Float64>("/docking/metric/elapsed_time", 5, true);

    ROS_INFO("[DockingMetric] Ready. Waiting for dock_pose and docked signal.");
  }

private:
  void dockCb(const geometry_msgs::PoseStamped::ConstPtr& msg)
  {
    if (!desired_set_)
    {
      desired_pose_ = *msg;
      desired_set_ = true;
      start_time_ = ros::Time::now();
      ROS_INFO("[DockingMetric] Desired pose recorded: (%.2f, %.2f) yaw=%.1f deg",
               msg->pose.position.x, msg->pose.position.y,
               tf2::getYaw(msg->pose.orientation) * 180.0 / M_PI);
    }
  }

  void dimsCb(const geometry_msgs::Vector3Stamped::ConstPtr& msg)
  {
    dock_length_ = msg->vector.x;
    dock_width_ = msg->vector.y;
  }

  void dockedCb(const std_msgs::Bool::ConstPtr& msg)
  {
    if (!msg->data || !desired_set_ || measured_) return;

    // Get robot pose
    geometry_msgs::TransformStamped tf;
    try {
      tf = tf_buffer_.lookupTransform(map_frame_, "base_link",
                                      ros::Time(0), ros::Duration(1.0));
    } catch (tf2::TransformException& ex) {
      ROS_WARN("[DockingMetric] TF failed: %s", ex.what());
      return;
    }

    double ax = tf.transform.translation.x;
    double ay = tf.transform.translation.y;
    double ayaw = tf2::getYaw(tf.transform.rotation);

    double dx = desired_pose_.pose.position.x;
    double dy = desired_pose_.pose.position.y;
    double dyaw = tf2::getYaw(desired_pose_.pose.orientation);

    double pos_err = std::hypot(ax - dx, ay - dy);
    double yaw_err = std::fabs(ayaw - dyaw);
    if (yaw_err > M_PI) yaw_err = 2 * M_PI - yaw_err;
    double yaw_err_deg = yaw_err * 180.0 / M_PI;

    double elapsed = (ros::Time::now() - start_time_).toSec();

    ROS_INFO("============================================");
    ROS_INFO("[DockingMetric] DOCKING COMPLETE");
    ROS_INFO("  Desired:  (%.2f, %.2f) yaw=%.1f deg", dx, dy, dyaw * 180.0 / M_PI);
    ROS_INFO("  Actual:   (%.2f, %.2f) yaw=%.1f deg", ax, ay, ayaw * 180.0 / M_PI);
    ROS_INFO("  Pos error:  %.2f m", pos_err);
    ROS_INFO("  Yaw error:  %.1f deg", yaw_err_deg);
    ROS_INFO("  Elapsed:    %.1f s", elapsed);
    if (dock_length_ > 0)
      ROS_INFO("  Dock size:  %.1f x %.1f m", dock_length_, dock_width_);
    ROS_INFO("============================================");

    std_msgs::Float64 pe, ye, te;
    pe.data = pos_err;
    ye.data = yaw_err_deg;
    te.data = elapsed;
    pos_err_pub_.publish(pe);
    yaw_err_pub_.publish(ye);
    time_pub_.publish(te);

    measured_ = true;
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber dock_sub_, docked_sub_, dims_sub_;
  ros::Publisher pos_err_pub_, yaw_err_pub_, time_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  geometry_msgs::PoseStamped desired_pose_;
  bool desired_set_ = false;
  bool measured_ = false;
  ros::Time start_time_;
  std::string map_frame_;
  double dock_length_ = 0, dock_width_ = 0;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "docking_metric_node");
  DockingMetric node;
  ros::spin();
  return 0;
}
