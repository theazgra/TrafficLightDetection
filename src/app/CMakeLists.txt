cmake_minimum_required(VERSION 3.8)
project(Bachelor)

include(../dlib/dlib/cmake)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp mmod_detector.cpp mmod_detector.h)

add_executable(Bachelor ${SOURCE_FILES})

target_link_libraries(Bachelor dlib::dlib)
