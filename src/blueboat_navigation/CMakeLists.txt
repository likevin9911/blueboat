cmake_minimum_required(VERSION 3.0.2)
project(blueboat_navigation)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2
  tf2_ros
  std_msgs
  nav_msgs
  geometry_msgs
  message_generation
  message_runtime
  steering_functions
  usv_msgs
  sensor_msgs
)

add_message_files(
  FILES
  DubinInput.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    tf2
    tf2_ros
    std_msgs
    nav_msgs
    geometry_msgs
    message_generation
    message_runtime
    steering_functions
    usv_msgs
    sensor_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## -----------------------------------------------------------------------------
## Nodes
## -----------------------------------------------------------------------------

# BINN-based coverage
add_executable(coverage_binn_node
  include/coverage_binn/coverage_binn.h
  include/coverage_binn/partition_binn.h
  include/coverage_binn/simple_dubins_path.h
  src/coverage_binn/coverage_binn_node.cpp
  src/coverage_binn/coverage_binn.cpp
  src/coverage_binn/partition_binn.cpp
  src/coverage_binn/simple_dubins_path_binn.cpp
)
add_dependencies(coverage_binn_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(coverage_binn_node ${catkin_LIBRARIES})

# Boustrophedon coverage with A* backtracking
add_executable(coverage_node
  include/coverage/a_star.h
  include/coverage/coverage.h
  include/coverage/partition.h
  include/simple_dubins_path/simple_dubins_path.h
  src/coverage_boustrophedon/simple_dubins_path.cpp
  src/coverage_boustrophedon/coverage_node.cpp
  src/coverage_boustrophedon/coverage.cpp
  src/coverage_boustrophedon/partition.cpp
)
add_dependencies(coverage_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(coverage_node ${catkin_LIBRARIES})

add_executable(simple_dubins_path_node
  include/simple_dubins_path/simple_dubins_path.h
  src/coverage_boustrophedon/simple_dubins_path_node.cpp
  src/coverage_boustrophedon/simple_dubins_path.cpp
)
add_dependencies(simple_dubins_path_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(simple_dubins_path_node ${catkin_LIBRARIES})

# Selected via ~planner_type param ("dubins" or "reeds_shepp").
add_executable(dubins_reeds_shepp_planner
  src/DubinsReedsSheppPlanner.cpp
)
add_dependencies(dubins_reeds_shepp_planner
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(dubins_reeds_shepp_planner ${catkin_LIBRARIES})

add_executable(dubins_planner
  src/steering/dubins_planner.cpp
)
add_dependencies(dubins_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dubins_planner
  ${catkin_LIBRARIES}
)
 
add_executable(cc_dubins_planner
  src/steering/cc_dubins_planner.cpp
)
add_dependencies(cc_dubins_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cc_dubins_planner
  ${catkin_LIBRARIES}
)
 
add_executable(reeds_shepp_planner
  src/steering/reeds_shepp_planner.cpp
)
add_dependencies(reeds_shepp_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(reeds_shepp_planner
  ${catkin_LIBRARIES}
)
 
add_executable(hc_rs_planner
  src/steering/hc_rs_planner.cpp
)
add_dependencies(hc_rs_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(hc_rs_planner
  ${catkin_LIBRARIES}
)

add_executable(hcpmpm_rs_planner
  src/steering/hcpmpm_rs_planner.cpp
)
add_dependencies(hcpmpm_rs_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(hcpmpm_rs_planner
  ${catkin_LIBRARIES}
)

#--- search planners----
add_executable(astar_planner       src/astar_planner.cpp)
add_dependencies(astar_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(astar_planner ${catkin_LIBRARIES})
 
add_executable(rrtstar_planner     src/rrtstar_planner.cpp)
add_dependencies(rrtstar_planner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(rrtstar_planner ${catkin_LIBRARIES})

add_executable(path_stitch        src/path_stitch.cpp)
add_dependencies(path_stitch ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(path_stitch ${catkin_LIBRARIES})
 
add_executable(vfh_node             src/vfh_node.cpp)
add_dependencies(vfh_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vfh_node ${catkin_LIBRARIES})



## Add custom (non compiling) targets so other files show up in QT Creator's project view.
file(GLOB_RECURSE EXTRA_FILES */*)
add_custom_target(${PROJECT_NAME}_OTHER_FILES ALL
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  SOURCES ${EXTRA_FILES}
)
