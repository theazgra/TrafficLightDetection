#ifndef DISPLAYIMAGE_OPENCVUTILS_H
#define DISPLAYIMAGE_OPENCVUTILS_H


#define int64 opencv_broken_int
#define uint64 opencv_broken_uint
#include <opencv2/opencv.hpp>
#undef int64
#undef uint64

#include <dlib/threads.h>
#include <dlib/dnn.h>
#include <exception>
#include <chrono>

struct ThreadParam2
{
    cv::Mat imgPart;
    float result;
};


struct TrafficLightPartInfo {
    cv::Mat trafficLightPart;
    std::vector<std::pair<cv::Scalar, cv::Scalar>> hsvRanges;
    cv::Mat1b resultMask;
    cv::Mat circleMask;
    float maskCoverage;
    int nonZeroPixelCount;

    TrafficLightPartInfo(cv::Mat & imgPart, std::vector<std::pair<cv::Scalar, cv::Scalar>> &hsvRanges)
    {
        this->trafficLightPart = imgPart;
        this->hsvRanges = hsvRanges;
        this->maskCoverage = 0.0f;
    }

};


enum TLState{
    Red = 0,
    Orange = 1,
    Green = 2,
    RedOrange = 3,
    Inactive = 4,
    Error = 5
};

std::string translate_TL_state(TLState state);
float get_mask_coverage(cv::Mat1b & mask);
void train_state_detection(const std::string file);
TLState get_traffic_light_state(cv::Mat & img);

void save_found_crop(cv::Mat & mat, dlib::mmod_rect rectangle, int index);




#endif //DISPLAYIMAGE_OPENCVUTILS_H

