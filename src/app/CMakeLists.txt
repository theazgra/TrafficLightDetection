cmake_minimum_required(VERSION 3.0)
project(tld)

include(../dlib/dlib/cmake)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA QUIET REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES extern_files/pugixml.cpp extern_files/args.hxx main.cpp net_definition.h settings.h settings.cpp traffic_light_train.cpp traffic_light_train.h
        cropper_test.cpp cropper_test.h traffic_light_test.cpp traffic_light_test.h cv_utils.cpp cv_utils.h
        Logger.cpp Logger.h Stopwatch.cpp Stopwatch.h main.h TrafficLightDetector.h location_detector_trainer.h state_detector_trainer.h)
#set(SOURCE_FILES main.cpp cropper_test.cpp cropper_test.h settings.h)

add_executable(tld ${SOURCE_FILES})

target_link_libraries(tld dlib::dlib)
target_link_libraries(tld ${OpenCV_LIBS})
