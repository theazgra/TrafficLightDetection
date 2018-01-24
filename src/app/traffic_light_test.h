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
    OnlyErrorDisplay
};


void test(std::string netFile, std::string testFile, TestType testType, bool saveImages);



#endif //BACHELOR_TRAFFIC_LIGHT_TEST_H
