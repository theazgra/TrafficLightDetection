#ifndef BACHELOR_TRAFFIC_LIGHT_TEST_H
#define BACHELOR_TRAFFIC_LIGHT_TEST_H

#include <iostream>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include "settings.h"
#include "Stopwatch.h"
#include "cv_utils.h"

enum TestType
{
    FullTest,
    DisplayOnly,
    NoDisplay,
    OnlyErrorDisplay
};

void test(std::string netFile, std::string testFile, TestType testType);
void save_detected_objects(const std::string netFile, const std::string xmlFile, const std::string folderPath, dlib::rectangle = dlib::rectangle());

void save_video(std::string netFile, std::string videoFile, std::string resultFolder);
void save_video_frames(std::string netFile, std::string xmlFile, std::string resultFolder);
void save_video_frames_with_sp(std::string netFile, std::string xmlFile, std::string resultFolder);

void visualize_detection(std::string netFile, std::string imgFile);

std::vector<std::vector<dlib::mmod_rect>> get_detected_rectanges(const std::string netFile, const std::string xmlFile);

int number_of_label_boxes(std::vector<dlib::mmod_rect> boxes);

TLState detect_state(const std::string netFile, const dlib::matrix<dlib::rgb_pixel> dlibImg);

TLState get_detected_state(const std::vector<dlib::mmod_rect>& detections, const dlib::matrix<dlib::rgb_pixel>& image);



#endif //BACHELOR_TRAFFIC_LIGHT_TEST_H
