cmake_minimum_required(VERSION 3.8)
project(XmlAnnotationInfoTool)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES main.cpp pugi_xml)
add_executable(XmlAnnotationInfoTool ${SOURCE_FILES})