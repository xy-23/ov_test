cmake_minimum_required(VERSION 3.10)

project(ov_test)

set(OpenVINO_DIR "/opt/intel/openvino_2023.3.0/runtime/cmake")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp)

target_link_libraries(main openvino::runtime ${OpenCV_LIBS})