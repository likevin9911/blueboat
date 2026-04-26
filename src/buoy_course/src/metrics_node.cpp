#include <buoy_course/metrics_node.h>

#include <std_msgs/String.h>
#include <std_msgs/Int32.h>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace buoy_course
{

MetricsNode::MetricsNode()
  : m_nhP("~")
  , m_tfListener(m_tfBuffer)
  , m_activeBuoyId(-1)
  , m_lapStarted(false)
  , m_lapFinished(false)
  , m_lapTime(0.0)
  , m_crossedStartOutbound(false)
  , m_hasPose(false)
{
  m_mapFrame          = m_nhP.param<std::string>("map_frame",        "map");
  m_roundingRadius    = m_nhP.param("rounding_radius",               8.0);
  m_logFile           = m_nhP.param<std::string>("log_file",         "/tmp/buoy_course_metrics.txt");
  m_startX            = m_nhP.param("start_x",                       0.0);
  m_startY            = m_nhP.param("start_y",                       0.0);
  m_startLineHalfWidth= m_nhP.param("start_line_half_width",         3.0);

  m_buoySub    = m_nh.subscribe("buoy_map",         10, &MetricsNode::buoyCb,    this);
  m_activeIdSub= m_nh.subscribe("active_buoy_id",   10, &MetricsNode::activeIdCb,this);
  m_statusPub  = m_nh.advertise<std_msgs::String>("course_metrics_status", 10, true);

  // Poll robot pose at 10 Hz
  m_updateTimer = m_nh.createTimer(ros::Duration(0.1), &MetricsNode::timerCb, this);

  ROS_INFO("[Metrics] Started. Log file: %s", m_logFile.c_str());
  ROS_INFO("[Metrics] Start position: (%.1f, %.1f)  Rounding radius: %.1f m",
           m_startX, m_startY, m_roundingRadius);
}

MetricsNode::~MetricsNode()
{
  if (m_lapStarted)
    printReport();
}

// ----------------------------------------------------------
void MetricsNode::buoyCb(const geometry_msgs::PoseArray::ConstPtr& msg)
{
  m_buoys = msg->poses;

  // Grow metrics vector to match confirmed buoys
  while (static_cast<int>(m_metrics.size()) < static_cast<int>(m_buoys.size()))
  {
    BuoyMetric bm;
    bm.buoy_id       = static_cast<int>(m_metrics.size()) + 1;
    bm.min_clearance = std::numeric_limits<double>::max();
    bm.max_clearance = 0.0;
    bm.sum_clearance = 0.0;
    bm.sample_count  = 0;
    m_metrics.push_back(bm);
  }
}

// ----------------------------------------------------------
void MetricsNode::activeIdCb(const std_msgs::Int32::ConstPtr& msg)
{
  m_activeBuoyId = msg->data;
}

// ----------------------------------------------------------
void MetricsNode::timerCb(const ros::TimerEvent&)
{
  // Get robot pose
  geometry_msgs::TransformStamped tf_stamped;
  try
  {
    tf_stamped = m_tfBuffer.lookupTransform(
        m_mapFrame, "base_link", ros::Time(0), ros::Duration(0.05));
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN_THROTTLE(5.0, "[Metrics] TF error: %s", ex.what());
    return;
  }

  double rx = tf_stamped.transform.translation.x;
  double ry = tf_stamped.transform.translation.y;

  // --- Start-line crossing ---
  if (!m_lapFinished)
  {
    bool crossed = checkStartLineCrossing(rx, ry);

    if (!m_lapStarted && crossed)
    {
      // First crossing = outbound lap start
      m_lapStarted          = true;
      m_crossedStartOutbound = true;
      m_lapStartTime        = ros::Time::now();
      ROS_INFO("[Metrics] *** LAP STARTED ***");
    }
    else if (m_lapStarted && m_crossedStartOutbound && crossed)
    {
      // Second crossing = lap complete
      m_lapEndTime  = ros::Time::now();
      m_lapTime     = (m_lapEndTime - m_lapStartTime).toSec();
      m_lapFinished = true;
      ROS_INFO("[Metrics] *** LAP COMPLETE ***  Time: %.2f s", m_lapTime);
      printReport();
      saveReport();
    }
  }

  // --- Clearance measurement ---
  if (m_lapStarted && !m_lapFinished)
    updateClearance(rx, ry);

  // --- Publish live status ---
  if (m_lapStarted)
  {
    double elapsed = m_lapFinished
        ? m_lapTime
        : (ros::Time::now() - m_lapStartTime).toSec();

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    if (m_lapFinished)
      ss << "LAP COMPLETE  Time: " << elapsed << " s\n";
    else
      ss << "LAP RUNNING   Elapsed: " << elapsed << " s\n";

    ss << "Active buoy: " << m_activeBuoyId << "\n";
    ss << "Confirmed buoys: " << m_buoys.size() << "\n\n";

    for (auto& bm : m_metrics)
    {
      ss << "Buoy " << bm.buoy_id << ":";
      if (bm.sample_count > 0)
        ss << "  min=" << bm.min_clearance
           << "m  avg=" << bm.avg_clearance()
           << "m  samples=" << bm.sample_count;
      else
        ss << "  (not yet rounded)";
      ss << "\n";
    }

    std_msgs::String status_msg;
    status_msg.data = ss.str();
    m_statusPub.publish(status_msg);
  }

  m_lastPose.position.x = rx;
  m_lastPose.position.y = ry;
  m_hasPose = true;
}

// ----------------------------------------------------------
void MetricsNode::updateClearance(double rx, double ry)
{
  // Measure clearance to ALL confirmed buoys, but only log it
  // under the metric for the currently active buoy to avoid
  // contaminating metrics with incidental proximity to other buoys.
  for (int i = 0; i < static_cast<int>(m_buoys.size()); i++)
  {
    double bx   = m_buoys[i].position.x;
    double by   = m_buoys[i].position.y;
    double dist = std::hypot(rx - bx, ry - by);

    // Only record when we're close enough to be "rounding" this buoy
    // AND it's the currently active buoy (set by course_planner)
    int buoy_id = i + 1;  // 1-based
    if (buoy_id != m_activeBuoyId) continue;
    if (dist > m_roundingRadius)   continue;

    if (i < static_cast<int>(m_metrics.size()))
    {
      BuoyMetric& bm = m_metrics[i];
      bm.min_clearance = std::min(bm.min_clearance, dist);
      bm.max_clearance = std::max(bm.max_clearance, dist);
      bm.sum_clearance += dist;
      bm.sample_count++;
    }
  }
}

// ----------------------------------------------------------
// Simple start-line crossing: virtual gate perpendicular to
// the course heading at the start position.
// Gate is a horizontal line segment at y = start_y,
// spanning [start_x - half_width, start_x + half_width].
// We detect crossing by checking if the robot just crossed y = start_y
// within x range.
bool MetricsNode::checkStartLineCrossing(double rx, double ry)
{
  if (!m_hasPose) return false;

  double prev_y = m_lastPose.position.y;
  double prev_x = m_lastPose.position.x;

  // Check if we crossed the start line (y = m_startY) in x range
  bool in_x_range = std::fabs(rx - m_startX) < m_startLineHalfWidth;
  bool crossed_y  = (prev_y - m_startY) * (ry - m_startY) < 0;  // sign change

  // Debounce: must be at least 5 s since last trigger
  static ros::Time last_trigger = ros::Time(0);
  if (ros::Time::now() - last_trigger < ros::Duration(5.0))
    return false;

  if (in_x_range && crossed_y)
  {
    last_trigger = ros::Time::now();
    ROS_INFO("[Metrics] Start-line crossing detected at (%.2f, %.2f)", rx, ry);
    return true;
  }
  return false;
}

// ----------------------------------------------------------
void MetricsNode::printReport()
{
  ROS_INFO("=================================================");
  ROS_INFO("[Metrics] COURSE REPORT");
  ROS_INFO("-------------------------------------------------");
  if (m_lapFinished)
    ROS_INFO("  Total lap time : %.3f s", m_lapTime);
  else
    ROS_INFO("  Lap time       : INCOMPLETE (%.3f s elapsed)",
             (ros::Time::now() - m_lapStartTime).toSec());

  ROS_INFO("  Buoys rounded  : %zu / %zu",
           std::count_if(m_metrics.begin(), m_metrics.end(),
               [](const BuoyMetric& b){ return b.sample_count > 0; }),
           m_metrics.size());
  ROS_INFO("-------------------------------------------------");

  for (auto& bm : m_metrics)
  {
    if (bm.sample_count > 0)
    {
      ROS_INFO("  Buoy %d | min=%.3f m | avg=%.3f m | max=%.3f m | n=%d",
               bm.buoy_id,
               bm.min_clearance,
               bm.avg_clearance(),
               bm.max_clearance,
               bm.sample_count);
    }
    else
    {
      ROS_INFO("  Buoy %d | NOT ROUNDED", bm.buoy_id);
    }
  }
  ROS_INFO("=================================================");
}

// ----------------------------------------------------------
void MetricsNode::saveReport()
{
  std::ofstream f(m_logFile);
  if (!f.is_open())
  {
    ROS_WARN("[Metrics] Could not open log file: %s", m_logFile.c_str());
    return;
  }

  f << std::fixed << std::setprecision(4);
  f << "=================================================\n";
  f << "BUOY COURSE METRICS REPORT\n";
  f << "Timestamp: " << ros::Time::now() << "\n";
  f << "-------------------------------------------------\n";
  f << "Total lap time (s): "
    << (m_lapFinished ? m_lapTime : (ros::Time::now() - m_lapStartTime).toSec())
    << (m_lapFinished ? "" : " (INCOMPLETE)") << "\n";
  f << "Buoys confirmed: " << m_buoys.size() << "\n";
  f << "Buoys rounded:   "
    << std::count_if(m_metrics.begin(), m_metrics.end(),
           [](const BuoyMetric& b){ return b.sample_count > 0; }) << "\n";
  f << "-------------------------------------------------\n";
  f << "buoy_id, min_clearance_m, avg_clearance_m, max_clearance_m, samples\n";
  for (auto& bm : m_metrics)
  {
    f << bm.buoy_id << ", ";
    if (bm.sample_count > 0)
      f << bm.min_clearance << ", "
        << bm.avg_clearance() << ", "
        << bm.max_clearance << ", "
        << bm.sample_count;
    else
      f << "N/A, N/A, N/A, 0";
    f << "\n";
  }
  f << "=================================================\n";
  f.close();

  ROS_INFO("[Metrics] Report saved to %s", m_logFile.c_str());
}

} // namespace buoy_course

int main(int argc, char** argv)
{
  ros::init(argc, argv, "metrics_node");
  buoy_course::MetricsNode node;
  ros::spin();
  return 0;
}
