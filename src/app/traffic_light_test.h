#ifndef BACHELOR_TRAFFIC_LIGHT_TEST_H
#define BACHELOR_TRAFFIC_LIGHT_TEST_H

#include <iostream>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include "settings.h"

enum TestType
{
    FullTest,
    DisplayOnly,
    NoDisplay,
    OnlyErrorDisplay,
    SaveCrops
};


void test(std::string netFile, std::string testFile, TestType testType, bool saveImages);

void save_video(std::string netFile, std::string videoFile, std::string resultFolder);
void save_video_frames(std::string netFile, std::string xmlFile, std::string resultFolder);
void save_video_frames_with_sp(std::string netFile, std::string xmlFile, std::string resultFolder);

int number_of_label_boxes(std::vector<dlib::mmod_rect> boxes);



#endif //BACHELOR_TRAFFIC_LIGHT_TEST_H
