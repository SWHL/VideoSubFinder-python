cmake_minimum_required(VERSION 3.20)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(${OpenCV_INCLUDE_DIRS})

add_executable(MyProject test_opencv.cpp)
target_link_libraries(MyProject ${OpenCV_LIBRARIES})
