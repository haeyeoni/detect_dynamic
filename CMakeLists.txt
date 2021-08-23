cmake_minimum_required(VERSION 3.0)
project(detect_dynamic)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  visualization_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  tf
)

catkin_package(
  DEPENDS PCL
)

include_directories(
    ${PCL_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}

    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR} 
)

add_executable(detect_dynamic main.cpp)

target_link_libraries(detect_dynamic
        ${PCL_LIBRARIES}
        ${catkin_LIBRARIES}
        ${Boost_LIBRARIES}
        ${Eigen3_LIBRARIES}
)