#ifndef DISPLAYIMAGE_OPENCVUTILS_H
#define DISPLAYIMAGE_OPENCVUTILS_H

//#define THREADING

#define int64 opencv_broken_int
#define uint64 opencv_broken_uint
#include <opencv2/opencv.hpp>
#undef int64
#undef uint64

#include <dlib/threads.h>
#include <dlib/data_io/load_image_dataset.h>
#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <exception>
#include "Stopwatch.h"
#include "Logger.h"
#include <cmath>

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
TLState get_traffic_light_state(cv::Mat & img, bool verbose = false);
TLState get_traffic_light_state2(dlib::matrix<dlib::rgb_pixel> dlibImg, bool verbose = false);

void save_found_crop(cv::Mat & mat, dlib::mmod_rect rectangle, int imgIndex, int labelIndex);
void save_found_crop(cv::Mat & mat, dlib::mmod_rect detRect, std::string fileName, dlib::rectangle sizeRect = dlib::rectangle());
cv::Mat crop_image(cv::Mat & mat, dlib::rectangle cropRectangle);

dlib::rgb_pixel get_color_for_state(TLState state);

bool valid_rectangle(const dlib::rectangle& rect, const cv::Mat& img);
bool valid_rectangle(const dlib::rectangle& rect, const dlib::matrix<dlib::rgb_pixel>& img);



#endif //DISPLAYIMAGE_OPENCVUTILS_H

