cmake_minimum_required(VERSION 3.0)
project(XmlAnnotationInfoTool)

#set(CMAKE_CXX_STANDARD 11)
set(CXX_STANDARD 11)

set(SOURCE_FILES main.cpp pugi_xml)
add_executable(XmlAnnotationInfoTool ${SOURCE_FILES})
