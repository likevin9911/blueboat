cmake_minimum_required(VERSION 3.0.2)
project(blueboat_control)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs geometry_msgs sensor_msgs usv_msgs tf2 tf2_ros)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(controller
  include/blueboat_control/controller.h
  src/controller.cpp
)
add_dependencies(controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(controller ${catkin_LIBRARIES})

add_executable(goal_to_path_node
  src/goal_to_path_node.cpp
)
add_dependencies(goal_to_path_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(goal_to_path_node ${catkin_LIBRARIES})
