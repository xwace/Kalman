cmake_minimum_required(VERSION 3.23)
project(Kalman)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

add_executable(Kalman main.cpp)
target_link_libraries(Kalman ${OpenCV_LIBS})