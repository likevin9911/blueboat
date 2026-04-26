cmake_minimum_required(VERSION 3.0.2)
project(dock_perception)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
  adaptive_clustering
)

find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS pcl_ros 
)

include_directories(
  include
  include/dock_perception
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(l_shape_fitting
  src/dock_pose_estimator.cpp
  src/lshape_estimator.cpp
)

add_dependencies(l_shape_fitting
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(
  l_shape_fitting
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

add_executable(dock_approach_planner
  src/dock_approach_planner.cpp
)
add_dependencies(dock_approach_planner
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(dock_approach_planner
  ${catkin_LIBRARIES}
)

add_executable(docking_metric_node src/docking_metric_node.cpp)
target_link_libraries(docking_metric_node ${catkin_LIBRARIES})
