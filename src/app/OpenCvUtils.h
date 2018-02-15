#ifndef DISPLAYIMAGE_OPENCVUTILS_H
#define DISPLAYIMAGE_OPENCVUTILS_H

//#define THREADING

#define int64 opencv_broken_int
#define uint64 opencv_broken_uint
#include <opencv2/opencv.hpp>
#undef int64
#undef uint64

#include <dlib/threads.h>
#include <exception>
#include <chrono>

struct GrayScaleTestParam
{
    cv::Mat imgPart;
    float result;
};


struct HsvTestParam {
    cv::Mat trafficLightPart;
    cv::Mat1b resultMask;
    std::vector<std::pair<cv::Scalar, cv::Scalar>> hsvRanges;
    float maskCoverage;

    HsvTestParam(cv::Mat & imgPart, std::vector<std::pair<cv::Scalar, cv::Scalar>> &hsvRanges)
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
TLState get_traffic_light_state(cv::Mat & img);

void save_found_crop(cv::Mat & mat, dlib::mmod_rect rectangle, int index);
cv::Mat crop_image(cv::Mat & mat, dlib::mmod_rect cropRectangle);




#endif //DISPLAYIMAGE_OPENCVUTILS_H

