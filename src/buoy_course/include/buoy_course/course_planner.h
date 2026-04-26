#pragma once

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/utils.h>

#include <vector>
#include <queue>
#include <limits>

namespace buoy_course
{

enum class CourseState
{
  WAITING_FOR_BUOYS,  // Discovering buoys, no circuit yet
  PLANNING,           // Circuit exists, planning A* approach
  NAVIGATING,         // Following A* approach to circuit entry
  ROUNDING,           // Following the closed-loop circuit (lapping)
  RETURNING,          // A* path back to start after all laps
  FINISHED            // Done
};

struct AStarNode
{
  int x, y;
  float g, h;
  int parent_x, parent_y;
  float f() const { return g + h; }
  bool operator>(const AStarNode& o) const { return f() > o.f(); }
};

class CoursePlanner
{
public:
  CoursePlanner();
  ~CoursePlanner() = default;

private:
  // ---- ROS handles ----
  ros::NodeHandle m_nh, m_nhP;
  ros::Subscriber m_buoySub, m_mapSub;
  ros::Publisher  m_pathPub, m_activeIdPub, m_statusPub;
  ros::Timer      m_planTimer;

  tf2_ros::Buffer            m_tfBuffer;
  tf2_ros::TransformListener m_tfListener;

  // ---- Parameters ----
  std::string m_mapFrame;
  double m_goalTolerance;
  double m_roundingOffset;
  double m_startX, m_startY;
  int    m_buoyCount;
  double m_planRate;

  // Circuit parameters
  int    m_totalLaps;            // Number of laps (-1 = infinite)
  bool   m_roundCW;              // true = CW (starboard), false = CCW (port)
  int    m_arcPoints;            // Arc interpolation points per buoy corner
  int    m_minBuoysForCircuit;   // Min buoys to start building circuit (default 2)
  int m_waypointsThisLap = 0;

  // ---- Runtime state ----
  CourseState m_state;
  std::vector<geometry_msgs::Pose> m_confirmedBuoys;
  std::vector<bool>                m_visited;
  nav_msgs::OccupancyGrid::ConstPtr m_map;

  int    m_currentMarkIdx;
  double m_robotX, m_robotY, m_robotPsi;
  geometry_msgs::PoseStamped m_currentGoal;

  bool   m_sequenceLocked;
  double m_courseCenterX, m_courseCenterY;

  // Circuit path (the closed-loop "raceline")
  std::vector<geometry_msgs::PoseStamped> m_circuitWaypoints;

  // Circuit following state
  int  m_currentWaypointIdx = 0;
  int  m_lapStartIdx        = 0;
  bool m_lapCompleted        = false;
  int  m_currentLap          = 0;
  int  m_lastBuoyCount       = 0;
  bool m_circuitDirty        = false;  // Set when circuit is rebuilt

  // ---- Callbacks ----
  void buoyCb(const geometry_msgs::PoseArray::ConstPtr& msg);
  void mapCb(const nav_msgs::OccupancyGrid::ConstPtr& msg);
  void timerCb(const ros::TimerEvent&);

  // ---- Sequence & circuit building ----
  void   sortBuoys();
  void   lockSequence();
  void   buildCircuit();
  void   densifyCircuit(double target_spacing);
  double estimateCircuitLength() const;

  // ---- FSM ----
  void        updateFSM(double rx, double ry, double psi);
  void        transitionTo(CourseState s);
  std::string stateName(CourseState s) const;

  // ---- Circuit following ----
  void publishCircuitPath();
  void advanceWaypointIndex(double rx, double ry);
  int  findNearestWaypoint(double rx, double ry) const;

  // ---- Legacy helpers ----
  int  nextUnvisitedIdx() const;
  int  visitedCount() const;

  // ---- A* path planning (approach & return only) ----
  bool  planPath(double sx, double sy, double gx, double gy,
                 nav_msgs::Path& path_out);
  bool  worldToGrid(double wx, double wy, int& gx, int& gy) const;
  bool  gridToWorld(int gx, int gy, double& wx, double& wy) const;
  bool  isFree(int gx, int gy) const;
  float heuristic(int ax, int ay, int bx, int by) const;
  void  publishActiveBuoyId(int id);
};

} // namespace buoy_course