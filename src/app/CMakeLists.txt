cmake_minimum_required(VERSION 3.0)
project(Bachelor)

include(../dlib/dlib/cmake)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES pugi_xml main.cpp net_definition.h settings.h settings.cpp traffic_light_train.cpp traffic_light_train.h
        cropper_test.cpp cropper_test.h utils.cpp utils.h traffic_light_test.cpp traffic_light_test.h OpenCvUtils.cpp OpenCvUtils.h
        Logger.cpp Logger.h Stopwatch.cpp Stopwatch.h)
#set(SOURCE_FILES main.cpp cropper_test.cpp cropper_test.h settings.h)

add_executable(Bachelor ${SOURCE_FILES})

target_link_libraries(Bachelor dlib::dlib)
target_link_libraries(Bachelor ${OpenCV_LIBS})
