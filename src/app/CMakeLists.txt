cmake_minimum_required(VERSION 3.0)
project(Bachelor)

include(../dlib/dlib/cmake)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp settings.h traffic_light_train.cpp traffic_light_train.h)
#set(SOURCE_FILES main.cpp mmod_detector.cpp mmod_detector.h car_detector_test.cpp car_detector.h cropper_test.cpp cropper_test.h settings.h traffic_light_train.cpp traffic_light_train.h)

add_executable(Bachelor ${SOURCE_FILES})

target_link_libraries(Bachelor dlib::dlib)
