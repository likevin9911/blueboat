#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <gazebo_msgs/ModelStates.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <mutex>
#include <random>
#include <vector>
#include <string>
#include <cmath>

class LiveLidarPostprocessNode {
public:
    LiveLidarPostprocessNode()
        : nh_(),
          pnh_("~"),
          gen_(std::random_device{}()) {
        pnh_.param<std::string>("input_lidar_topic", input_lidar_topic_,
                                "/blueboat/sensors/lidars/lidar_blueboat/points");
        pnh_.param<std::string>("input_camera_topic", input_camera_topic_,
                                "/blueboat/sensors/cameras/front_camera/blueboat/sensors/cameras/front_camera/image_raw");
        pnh_.param<std::string>("input_model_states_topic", input_model_states_topic_,
                                "/gazebo/model_states");
        pnh_.param<std::string>("output_lidar_topic", output_lidar_topic_,
                                "/blueboat/sensors/lidars/lidar_blueboat/points_noisy_live");
        pnh_.param<std::string>("output_debug_topic", output_debug_topic_,
                                "/blueboat/sensors/lidars/lidar_blueboat/points_noisy_debug");
        pnh_.param<std::string>("frame_id_override", frame_id_override_, "");

        pnh_.param<bool>("use_camera", use_camera_, true);
        pnh_.param<bool>("use_model_states", use_model_states_, true);

        pnh_.param<double>("range_noise_stddev", range_noise_stddev_, 0.03);
        pnh_.param<double>("xy_noise_stddev", xy_noise_stddev_, 0.01);
        pnh_.param<double>("z_noise_stddev", z_noise_stddev_, 0.01);

        pnh_.param<double>("drop_probability", drop_probability_, 0.02);
        pnh_.param<double>("noise_point_probability", noise_point_probability_, 0.01);
        pnh_.param<int>("max_noise_points", max_noise_points_, 200);

        pnh_.param<double>("max_range", max_range_, 100.0);
        pnh_.param<double>("camera_timeout", camera_timeout_, 0.5);
        pnh_.param<double>("process_rate_limit", process_rate_limit_, 0.0);

        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_lidar_topic_, 1);
        debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_debug_topic_, 1);

        lidar_sub_ = nh_.subscribe(input_lidar_topic_, 1,
                                   &LiveLidarPostprocessNode::lidarCallback, this);

        if (use_camera_) {
            camera_sub_ = nh_.subscribe(input_camera_topic_, 1,
                                        &LiveLidarPostprocessNode::cameraCallback, this);
        }

        if (use_model_states_) {
            model_states_sub_ = nh_.subscribe(input_model_states_topic_, 1,
                                              &LiveLidarPostprocessNode::modelStatesCallback, this);
        }

        ROS_INFO("LiveLidarPostprocessNode started");
        ROS_INFO(" input lidar: %s", input_lidar_topic_.c_str());
        ROS_INFO(" input camera: %s", input_camera_topic_.c_str());
        ROS_INFO(" input model states: %s", input_model_states_topic_.c_str());
        ROS_INFO(" output lidar: %s", output_lidar_topic_.c_str());
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber lidar_sub_;
    ros::Subscriber camera_sub_;
    ros::Subscriber model_states_sub_;

    ros::Publisher cloud_pub_;
    ros::Publisher debug_pub_;

    std::string input_lidar_topic_;
    std::string input_camera_topic_;
    std::string input_model_states_topic_;
    std::string output_lidar_topic_;
    std::string output_debug_topic_;
    std::string frame_id_override_;

    bool use_camera_;
    bool use_model_states_;

    double range_noise_stddev_;
    double xy_noise_stddev_;
    double z_noise_stddev_;
    double drop_probability_;
    double noise_point_probability_;
    int max_noise_points_;
    double max_range_;
    double camera_timeout_;
    double process_rate_limit_;

    std::mutex data_mutex_;
    sensor_msgs::ImageConstPtr latest_image_;
    gazebo_msgs::ModelStatesConstPtr latest_model_states_;
    ros::Time last_processed_time_;

    std::mt19937 gen_;

    void cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_image_ = msg;
    }

    void modelStatesCallback(const gazebo_msgs::ModelStatesConstPtr& msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        latest_model_states_ = msg;
    }

    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (process_rate_limit_ > 0.0) {
            if (!last_processed_time_.isZero()) {
                double dt = (msg->header.stamp - last_processed_time_).toSec();
                if (dt >= 0.0 && dt < (1.0 / process_rate_limit_)) {
                    return;
                }
            }
            last_processed_time_ = msg->header.stamp;
        }

        sensor_msgs::ImageConstPtr image_msg;
        gazebo_msgs::ModelStatesConstPtr model_states_msg;

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            image_msg = latest_image_;
            model_states_msg = latest_model_states_;
        }

        if (use_camera_) {
            if (!image_msg) {
                ROS_WARN_THROTTLE(2.0, "Waiting for camera image");
                return;
            }
            if (!image_msg->header.stamp.isZero() && !msg->header.stamp.isZero()) {
                if (std::fabs((msg->header.stamp - image_msg->header.stamp).toSec()) > camera_timeout_) {
                    ROS_WARN_THROTTLE(2.0, "Camera image too old relative to LiDAR");
                    return;
                }
            }
        }

        pcl::PointCloud<pcl::PointXYZI> in_cloud;
        pcl::fromROSMsg(*msg, in_cloud);

        if (in_cloud.empty()) {
            ROS_WARN_THROTTLE(2.0, "Received empty cloud");
            return;
        }

        pcl::PointCloud<pcl::PointXYZI> out_cloud;
        out_cloud.header = in_cloud.header;
        out_cloud.is_dense = false;
        out_cloud.points.reserve(in_cloud.points.size() + static_cast<size_t>(max_noise_points_));

        std::normal_distribution<float> n_xy(0.0f, static_cast<float>(xy_noise_stddev_));
        std::normal_distribution<float> n_z(0.0f, static_cast<float>(z_noise_stddev_));
        std::normal_distribution<float> n_range(0.0f, static_cast<float>(range_noise_stddev_));
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        std::uniform_real_distribution<float> u_range_x(-static_cast<float>(max_range_), static_cast<float>(max_range_));
        std::uniform_real_distribution<float> u_range_y(-static_cast<float>(max_range_), static_cast<float>(max_range_));
        std::uniform_real_distribution<float> u_range_z(-2.0f, 2.0f);
        std::uniform_real_distribution<float> u_intensity(0.0f, 255.0f);

        cv::Mat cv_image;
        if (use_camera_) {
            try {
                cv_bridge::CvImageConstPtr cv_ptr =
                    cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8);
                cv_image = cv_ptr->image;
            } catch (const cv_bridge::Exception& e) {
                ROS_ERROR_THROTTLE(2.0, "cv_bridge failed: %s", e.what());
                return;
            }
        }

        for (const auto& p : in_cloud.points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                continue;
            }

            if (u01(gen_) < static_cast<float>(drop_probability_)) {
                continue;
            }

            pcl::PointXYZI q = p;

            float r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (r > 1e-4f) {
                float dr = n_range(gen_);
                float scale = std::max(0.0f, (r + dr) / r);
                q.x = p.x * scale + n_xy(gen_);
                q.y = p.y * scale + n_xy(gen_);
                q.z = p.z * scale + n_z(gen_);
            } else {
                q.x += n_xy(gen_);
                q.y += n_xy(gen_);
                q.z += n_z(gen_);
            }

            if (use_camera_ && !cv_image.empty()) {
                // Placeholder for camera-aware adjustment.
                // Later you can project points and alter intensity/class-based noise.
                q.intensity = p.intensity;
            } else {
                q.intensity = p.intensity;
            }

            out_cloud.points.push_back(q);
        }

        int injected = 0;
        for (size_t i = 0; i < in_cloud.points.size(); ++i) {
            if (injected >= max_noise_points_) {
                break;
            }
            if (u01(gen_) < static_cast<float>(noise_point_probability_)) {
                pcl::PointXYZI n;
                n.x = u_range_x(gen_);
                n.y = u_range_y(gen_);
                n.z = u_range_z(gen_);
                n.intensity = u_intensity(gen_);
                out_cloud.points.push_back(n);
                injected++;
            }
        }

        out_cloud.width = static_cast<uint32_t>(out_cloud.points.size());
        out_cloud.height = 1;

        sensor_msgs::PointCloud2 out_msg;
        pcl::toROSMsg(out_cloud, out_msg);
        out_msg.header = msg->header;

        if (!frame_id_override_.empty()) {
            out_msg.header.frame_id = frame_id_override_;
        }

        cloud_pub_.publish(out_msg);
        debug_pub_.publish(*msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "live_lidar_postprocess_node");
    LiveLidarPostprocessNode node;
    ros::spin();
    return 0;
}
