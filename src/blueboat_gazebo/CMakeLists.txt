cmake_minimum_required(VERSION 3.0.2)
project(blueboat_gazebo)

find_package(catkin REQUIRED COMPONENTS
  blueboat_description
  gazebo_dev
  gazebo_msgs
  gazebo_ros
  geographic_msgs
  message_filters
  message_generation
  roscpp
  rospy
  robot_localization
  std_msgs
  urdf
  usv_gazebo_plugins
  wave_gazebo
  xacro
  vrx_gazebo
)

catkin_package(
  CATKIN_DEPENDS
    blueboat_description
    gazebo_dev
    gazebo_msgs
    gazebo_ros
    geographic_msgs
    message_filters
    message_runtime
    roscpp
    rospy
    robot_localization
    std_msgs
    urdf
    usv_gazebo_plugins
    wave_gazebo
    xacro
    
    vrx_gazebo
)

# Register env-hook so GAZEBO_MODEL_PATH is set on source setup.bash
catkin_add_env_hooks(99_blueboat_gazebo
  SHELLS sh
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)

# Generate urdf from xacro and install
xacro_add_files(
  urdf/blueboat_gazebo.urdf.xacro
  INSTALL DESTINATION urdf)

install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models)

include_directories(include ${catkin_INCLUDE_DIRS})

## Build GPU Velodyne plugin
add_library(gazebo_ros_velodyne_gpu_laser SHARED
  src/GazeboRosVelodyneLaser.cpp
)
target_compile_definitions(gazebo_ros_velodyne_gpu_laser PRIVATE
  GAZEBO_GPU_RAY=1
)
target_include_directories(gazebo_ros_velodyne_gpu_laser PUBLIC
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)
target_link_libraries(gazebo_ros_velodyne_gpu_laser
  /usr/lib/x86_64-linux-gnu/gazebo-11/plugins/libGpuRayPlugin.so
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)
install(TARGETS gazebo_ros_velodyne_gpu_laser
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
