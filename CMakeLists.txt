cmake_minimum_required(VERSION 3.0.2)
project(image_enhance_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


add_executable(image_enhance_node enhance_node/src/enhancement.cpp enhance_node/src/enhancer/enhancer.cpp)
target_link_libraries(image_enhance_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

