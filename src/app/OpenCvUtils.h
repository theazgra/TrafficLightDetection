#include <opencv2/opencv.hpp>
#include <dlib/threads.h>
#include <exception>
#include <chrono>

#ifndef DISPLAYIMAGE_OPENCVUTILS_H
#define DISPLAYIMAGE_OPENCVUTILS_H

struct TrafficLightPartInfo{
    cv::Mat trafficLightPart;
    cv::Scalar lowerHSVBound;
    cv::Scalar upperHSBBound;
    cv::Mat1b resultMask;
    float maskCoverage;
};

struct Color{
    int red;
    int green;
    int blue;

    int combined()
    {
        return red + green + blue;
    }

};

enum TLState{
    Red,
    Orange,
    Green,
    Inactive
};


//Color get_average_color(cv::Mat & img);
//float get_brightness_value(cv::Mat1b & maskedImg);

//TLState get_traffic_light_state(cv::Mat & img);
TLState get_traffic_light_state();



#endif //DISPLAYIMAGE_OPENCVUTILS_H

