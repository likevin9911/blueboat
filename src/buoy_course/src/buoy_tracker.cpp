#include <buoy_course/buoy_tracker.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>

#include <cmath>
#include <limits>

namespace buoy_course
{

// ============================================================
// TrackedBuoy — constant-velocity Kalman filter
// State vector: [x, y, vx, vy]
// ============================================================

TrackedBuoy::TrackedBuoy()
  : id(-1), x(0), y(0), vx(0), vy(0)
  , hit_count(0), miss_count(0), confirmed(false), active(true)
{
  state = {0, 0, 0, 0};
  // Initial covariance: high uncertainty on position, very high on velocity
  P = {
    4.0, 0.0, 0.0, 0.0,
    0.0, 4.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0
  };
}

// Constant-velocity prediction step
void TrackedBuoy::predict(double dt)
{
  // F = [ 1 0 dt  0 ]
  //     [ 0 1  0 dt ]
  //     [ 0 0  1  0 ]
  //     [ 0 0  0  1 ]
  double new_x  = state[0] + state[2] * dt;
  double new_y  = state[1] + state[3] * dt;
  double new_vx = state[2];
  double new_vy = state[3];

  state = {new_x, new_y, new_vx, new_vy};
  x = new_x;
  y = new_y;
  vx = new_vx;
  vy = new_vy;

  // Process noise Q (buoys are stationary — keep velocity noise small)
  double q_pos = 0.01 * dt * dt;  // position process noise
  double q_vel = 0.05 * dt;       // velocity process noise

  // P = F*P*F' + Q  (simplified for const-vel model)
  // Update position variance
  P[0]  += 2*dt*P[2]  + dt*dt*P[10] + q_pos;  // P[0,0]
  P[5]  += 2*dt*P[7]  + dt*dt*P[15] + q_pos;  // P[1,1]
  P[10] += q_vel;                               // P[2,2]
  P[15] += q_vel;                               // P[3,3]
  // Off-diagonals (position-velocity cross terms)
  P[2]  += dt * P[10];   // P[0,2]
  P[8]  = P[2];          // P[2,0]
  P[7]  += dt * P[15];   // P[1,3]
  P[13] = P[7];          // P[3,1]
}

// Measurement update: observe (x, y) directly
void TrackedBuoy::update(double meas_x, double meas_y, double meas_noise)
{
  // H = [1 0 0 0]
  //     [0 1 0 0]
  // R = meas_noise^2 * I_2

  double R = meas_noise * meas_noise;

  // Innovation
  double innov_x = meas_x - state[0];
  double innov_y = meas_y - state[1];

  // S = H*P*H' + R  =>  S_xx = P[0,0]+R, S_yy = P[1,1]+R
  double S_xx = P[0]  + R;
  double S_yy = P[5]  + R;

  // Kalman gain K = P*H' * S^-1
  // K is 4x2; only need K[:,0] and K[:,1]
  double K0x = P[0]  / S_xx;   // K[0,x]
  double K1x = P[4]  / S_xx;   // K[1,x]  (= P[1,0]/S_xx)
  double K2x = P[8]  / S_xx;   // K[2,x]
  double K3x = P[12] / S_xx;   // K[3,x]

  double K0y = P[1]  / S_yy;   // K[0,y]
  double K1y = P[5]  / S_yy;   // K[1,y]
  double K2y = P[9]  / S_yy;   // K[2,y]
  double K3y = P[13] / S_yy;   // K[3,y]

  // State update
  state[0] += K0x * innov_x + K0y * innov_y;
  state[1] += K1x * innov_x + K1y * innov_y;
  state[2] += K2x * innov_x + K2y * innov_y;
  state[3] += K3x * innov_x + K3y * innov_y;

  x  = state[0];
  y  = state[1];
  vx = state[2];
  vy = state[3];

  // Covariance update  P = (I - K*H) * P
  // Only update the rows/cols touched by H (rows 0 and 1 of K)
  P[0]  -= K0x * P[0]  + K0y * P[4];
  P[1]  -= K0x * P[1]  + K0y * P[5];
  P[2]  -= K0x * P[2]  + K0y * P[6];
  P[3]  -= K0x * P[3]  + K0y * P[7];

  P[4]  -= K1x * P[0]  + K1y * P[4];
  P[5]  -= K1x * P[1]  + K1y * P[5];
  P[6]  -= K1x * P[2]  + K1y * P[6];
  P[7]  -= K1x * P[3]  + K1y * P[7];

  P[8]  -= K2x * P[0]  + K2y * P[4];
  P[9]  -= K2x * P[1]  + K2y * P[5];
  P[10] -= K2x * P[2]  + K2y * P[6];
  P[11] -= K2x * P[3]  + K2y * P[7];

  P[12] -= K3x * P[0]  + K3y * P[4];
  P[13] -= K3x * P[1]  + K3y * P[5];
  P[14] -= K3x * P[2]  + K3y * P[6];
  P[15] -= K3x * P[3]  + K3y * P[7];
}

// ============================================================
// BuoyTracker node
// ============================================================

BuoyTracker::BuoyTracker()
  : m_nhP("~")
  , m_tfListener(m_tfBuffer)
  , m_nextTrackId(1)
{
  // Parameters
  m_assocRadius   = m_nhP.param("association_radius",   2.0);
  m_measNoise     = m_nhP.param("measurement_noise",    0.3);
  m_confirmThresh = m_nhP.param("confirmation_count",   8);
  m_maxMissCount  = m_nhP.param("max_miss_count",       10);
  m_maxBuoys      = m_nhP.param("buoy_count",           4);
  m_maxRange      = m_nhP.param("max_detection_range",  30.0);
  m_minRange      = m_nhP.param("min_detection_range",  1.0);
  m_mapFrame      = m_nhP.param<std::string>("map_frame", "map");

  // The adaptive_clustering package publishes clusters as individual
  // PointCloud2 topics under /adaptive_clustering/clusters
  // Each message is one cluster; we compute its centroid.
  m_clusterSub = m_nh.subscribe(
    "adaptive_clustering/clusters", 100, &BuoyTracker::clusterCb, this);

  m_buoyPub   = m_nh.advertise<geometry_msgs::PoseArray>("buoy_map", 10, true);
  m_markerPub = m_nh.advertise<visualization_msgs::MarkerArray>(
      "buoy_markers", 10, true);

  m_lastUpdateTime = ros::Time::now();

  ROS_INFO("[BuoyTracker] Started. association_radius=%.2f  confirm_thresh=%d",
           m_assocRadius, m_confirmThresh);
}

// ----------------------------------------------------------
// Cluster callback — one call per cluster per scan
// ----------------------------------------------------------
void BuoyTracker::clusterCb(const adaptive_clustering::ClusterArray::ConstPtr& msg)
{
  if (msg->clusters.empty()) return;

  ros::Time now = msg->clusters[0].header.stamp;
  double dt = (now - m_lastUpdateTime).toSec();
  if (dt <= 0.0 || dt > 2.0) dt = 0.1;

  predictAll(dt);
  m_lastUpdateTime = now;

  // Get robot position for range filter
  geometry_msgs::TransformStamped robot_tf;
  try
  {
    robot_tf = m_tfBuffer.lookupTransform(
        m_mapFrame, "base_link", ros::Time(0), ros::Duration(0.1));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN_THROTTLE(5.0, "[BuoyTracker] Robot TF error: %s", ex.what());
    return;
  }
  double rx = robot_tf.transform.translation.x;
  double ry = robot_tf.transform.translation.y;

  for (const auto& cluster_cloud : msg->clusters)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(cluster_cloud, cloud);
    if (cloud.empty()) continue;

    Eigen::Vector4f centroid_sensor;
    pcl::compute3DCentroid(cloud, centroid_sensor);

    geometry_msgs::PointStamped pt_in, pt_out;
    pt_in.header = cluster_cloud.header;
    if (pt_in.header.frame_id.empty())
    pt_in.header.frame_id = "lidar_blueboat_link";
    if (pt_in.header.stamp.isZero())
    pt_in.header.stamp = ros::Time::now();
    
    pt_in.point.x = centroid_sensor[0];
    pt_in.point.y = centroid_sensor[1];
    pt_in.point.z = centroid_sensor[2];

    try
    {
      m_tfBuffer.transform(pt_in, pt_out, m_mapFrame, ros::Duration(0.1));
    }
    catch (tf2::TransformException& ex)
    {
      ROS_WARN_THROTTLE(5.0, "[BuoyTracker] TF error: %s", ex.what());
      continue;
    }

    double mx = pt_out.point.x;
    double my = pt_out.point.y;

    double range = std::hypot(mx - rx, my - ry);
    if (range < m_minRange || range > m_maxRange) continue;

    int best = findBestTrack(mx, my);
    if (best >= 0)
    {
      m_tracks[best].update(mx, my, m_measNoise);
      m_tracks[best].hit_count++;
      m_tracks[best].miss_count = 0;
      m_tracks[best].last_seen  = now;

      if (!m_tracks[best].confirmed &&
          m_tracks[best].hit_count >= m_confirmThresh)
      {
        int confirmed_count = 0;
        for (auto& t : m_tracks)
          if (t.confirmed) confirmed_count++;

        if (confirmed_count < m_maxBuoys)
        {
          m_tracks[best].confirmed = true;
          m_tracks[best].id        = m_nextTrackId++;
          ROS_INFO("[BuoyTracker] Buoy %d confirmed at (%.2f, %.2f)",
                   m_tracks[best].id, m_tracks[best].x, m_tracks[best].y);
        }
      }
    }
    else
    {
      if (static_cast<int>(m_tracks.size()) < m_maxBuoys * 3)
        createNewTrack(mx, my);
    }
  }

  removeDeadTracks();
  publishBuoys();
  publishMarkers();
}
// ----------------------------------------------------------
void BuoyTracker::predictAll(double dt)
{
  for (auto& t : m_tracks)
    if (t.active)
    {
      t.predict(dt);
      t.miss_count++;
    }
}

// ----------------------------------------------------------
int BuoyTracker::findBestTrack(double x, double y)
{
  int    best_idx  = -1;
  double best_dist = m_assocRadius;

  for (int i = 0; i < static_cast<int>(m_tracks.size()); i++)
  {
    if (!m_tracks[i].active) continue;
    double d = std::hypot(x - m_tracks[i].x, y - m_tracks[i].y);
    if (d < best_dist)
    {
      best_dist = d;
      best_idx  = i;
    }
  }
  return best_idx;
}

// ----------------------------------------------------------
void BuoyTracker::createNewTrack(double x, double y)
{
  TrackedBuoy t;
  t.state    = {x, y, 0.0, 0.0};
  t.x        = x;
  t.y        = y;
  t.hit_count = 1;
  t.miss_count = 0;
  t.last_seen = m_lastUpdateTime;
  m_tracks.push_back(t);
}

// ----------------------------------------------------------
void BuoyTracker::removeDeadTracks()
{
  m_tracks.erase(
    std::remove_if(m_tracks.begin(), m_tracks.end(),
      [this](const TrackedBuoy& t) {
        // Never remove a confirmed buoy — its position is ground truth
        if (t.confirmed) return false;
        return t.miss_count > m_maxMissCount;
      }),
    m_tracks.end());
}

// ----------------------------------------------------------
void BuoyTracker::publishBuoys()
{
  geometry_msgs::PoseArray msg;
  msg.header.stamp    = ros::Time::now();
  msg.header.frame_id = m_mapFrame;

  // Publish confirmed buoys sorted by ID so consumers get stable ordering
  std::vector<TrackedBuoy*> confirmed;
  for (auto& t : m_tracks)
    if (t.confirmed) confirmed.push_back(&t);

  std::sort(confirmed.begin(), confirmed.end(),
            [](TrackedBuoy* a, TrackedBuoy* b){ return a->id < b->id; });

  for (auto* t : confirmed)
  {
    geometry_msgs::Pose p;
    p.position.x = t->x;
    p.position.y = t->y;
    p.position.z = 0.0;
    p.orientation.w = 1.0;
    msg.poses.push_back(p);
  }

  m_buoyPub.publish(msg);
}

// ----------------------------------------------------------
void BuoyTracker::publishMarkers()
{
  visualization_msgs::MarkerArray ma;
  ros::Time now = ros::Time::now();

  // Delete-all marker to clear stale tracks
  visualization_msgs::Marker del;
  del.action = visualization_msgs::Marker::DELETEALL;
  del.ns = "buoy_tracker";
  ma.markers.push_back(del);

  for (auto& t : m_tracks)
  {
    if (!t.active) continue;

    visualization_msgs::Marker m;
    m.header.frame_id = m_mapFrame;
    m.header.stamp    = now;
    m.ns              = "buoy_tracker";
    m.id              = t.id >= 0 ? t.id : (1000 + static_cast<int>(&t - &m_tracks[0]));
    m.type            = visualization_msgs::Marker::CYLINDER;
    m.action          = visualization_msgs::Marker::ADD;
    m.pose.position.x = t.x;
    m.pose.position.y = t.y;
    m.pose.position.z = 0.5;
    m.pose.orientation.w = 1.0;
    m.scale.x = 0.6;
    m.scale.y = 0.6;
    m.scale.z = 1.0;
    m.lifetime = ros::Duration(1.0);

    if (t.confirmed)
    {
      // Confirmed — bright orange/yellow (no color sensor, so neutral)
      m.color.r = 1.0f; m.color.g = 0.6f; m.color.b = 0.0f; m.color.a = 1.0f;

      // Label
      visualization_msgs::Marker label = m;
      label.id   = m.id + 500;
      label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      label.text = "Buoy " + std::to_string(t.id);
      label.pose.position.z = 1.4;
      label.scale.z = 0.5;
      label.color.r = 1.0f;
      label.color.g = 1.0f;
      label.color.b = 1.0f;
      label.color.a = 1.0f;
      ma.markers.push_back(label);
    }
    else
    {
      // Unconfirmed — grey, smaller
      m.color.r = 0.5f; m.color.g = 0.5f; m.color.b = 0.5f; m.color.a = 0.5f;
      m.scale.x = 0.3; m.scale.y = 0.3;
    }

    ma.markers.push_back(m);
  }

  m_markerPub.publish(ma);
}



} // namespace buoy_course
int main(int argc, char** argv)
{
  ros::init(argc, argv, "buoy_tracker_node");
  buoy_course::BuoyTracker node;
  ros::spin();
  return 0;
}
