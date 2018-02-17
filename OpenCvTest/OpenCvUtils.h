#include <opencv2/opencv.hpp>
#include <chrono>

#ifndef DISPLAYIMAGE_OPENCVUTILS_H
#define DISPLAYIMAGE_OPENCVUTILS_H


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


Color get_average_color(cv::Mat & img);
float get_average_brightness(cv::Mat & img);

TLState get_traffic_light_state(cv::Mat & img);



#endif //DISPLAYIMAGE_OPENCVUTILS_H

