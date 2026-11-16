cmake_minimum_required(VERSION 3.16)
project(ml_media_processing CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
# HailoRT ставится вместе с DFC/HailoRT runtime; в примерах Hailo используется
# find_package(HailoRT 5 REQUIRED) — версию не фиксируем, чтобы подошла и 4.x.
find_package(HailoRT REQUIRED)

add_executable(yolo26s_seg yolo26s_seg.cpp)
target_link_libraries(yolo26s_seg PRIVATE ${OpenCV_LIBS} HailoRT::libhailort)

add_executable(yolo11s_seg yolo11s_seg.cpp)
target_link_libraries(yolo11s_seg PRIVATE ${OpenCV_LIBS} HailoRT::libhailort)
