cmake_minimum_required(VERSION 3.8)
project(OpenCvTest)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp OpenCvUtils.cpp OpenCvUtils.h)
add_executable(OpenCvTest ${SOURCE_FILES})

cmake_minimum_required(VERSION 2.8)

target_link_libraries( OpenCvTest ${OpenCV_LIBS})