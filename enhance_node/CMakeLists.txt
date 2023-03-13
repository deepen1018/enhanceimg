cmake_minimum_required(VERSION 2.8.3)
project(ie_project)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  cv_bridge
  roscpp
  std_msgs
  image_transport
)
    
find_package(OpenCV 4 REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

catkin_package()
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(ie_node src/enhancement.cpp)
target_link_libraries(ie_node ${OpenCV_LIBS} ${catkin_LIBRARIES}) 
