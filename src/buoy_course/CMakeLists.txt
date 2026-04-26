cmake_minimum_required(VERSION 3.0.2)
project(buoy_course)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  adaptive_clustering
)

find_package(PCL 1.7 REQUIRED COMPONENTS common)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES buoy_course
  CATKIN_DEPENDS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    pcl_ros
    pcl_conversions
    adaptive_clustering
)

include_directories(
  include
  include/buoy_course
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# ── buoy_tracker_node ──
add_executable(buoy_tracker_node
  src/buoy_tracker.cpp
)
target_link_libraries(buoy_tracker_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# ── course_planner_node ──
add_executable(course_planner_node
  src/course_planner.cpp
)
target_link_libraries(course_planner_node
  ${catkin_LIBRARIES}
)

# ── metrics_node ──
add_executable(metrics_node
  src/metrics_node.cpp
)
target_link_libraries(metrics_node
  ${catkin_LIBRARIES}
)

install(TARGETS buoy_tracker_node course_planner_node metrics_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
