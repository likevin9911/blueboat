cmake_minimum_required(VERSION 3.1)
project(map_inflating)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  costmap_2d
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    sensor_msgs
    costmap_2d
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  OPTIONAL
)
