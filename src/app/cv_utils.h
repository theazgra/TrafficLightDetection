/** \file cv_utils.h
 * Functions using both OpenCV and dlib libraries to perform operations on images.
 *
 */


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
#include <dlib/algs.h>
#include <exception>
#include "Stopwatch.h"
#include "Logger.h"
#include <cmath>

/// Structure used to pass parameters to thread for grayscale test.
struct GrayScaleTestParam
{
    /// Image part which to test.
    cv::Mat imgPart;
    /// Average brightness in image given part.
    float result;
};

/// Structure used to pass parameters to thread for HSV test.
struct HsvTestParam {
    /// Image part which to test.
    cv::Mat trafficLightPart;

    ///Resulting mask.
    cv::Mat1b resultMask;

    /// HSV ranges to test (mask).
    std::vector<std::pair<cv::Scalar, cv::Scalar>> hsvRanges;

    /// Percentage coverage of image masked by HSV ranges.
    float maskCoverage;

    /// Constructor, add ranges and initialize mask coverage to 0.0
    /// \param imgPart Image part which to test.
    /// \param hsvRanges HSV ranges to mask.
    HsvTestParam(cv::Mat & imgPart, std::vector<std::pair<cv::Scalar, cv::Scalar>> &hsvRanges)
    {
        this->trafficLightPart = imgPart;
        this->hsvRanges = hsvRanges;
        this->maskCoverage = 0.0f;
    }

};

/// State of traffic light.
enum TLState{
    /// Stop.
    Red = 0,
    /// You should stop.
    Orange = 1,
    /// Go.
    Green = 2,
    /// Prepare to go.
    RedOrange = 3,
    /// Traffic light is inactive.
    Inactive = 4,
    /// Traffic light state couldn't be detected.
    Ambiguous = 5
};

/// Show image in window (OpenCV version).
/// \param img OpenCV image to show.
/// \param winName Window name.
void show(cv::Mat & img, std::string winName = "");

/// Show image in window (Dlib version).
/// \param image Dlib image, currently only dlib::matrix<dib::rgb_pixel>
void show(dlib::matrix<dlib::rgb_pixel>& image);

/// Check if all values are equal or below threshold value.
/// \param values Vector of float values.
/// \param threshold Threshold value.
/// \return True if all values are equal or smaller than threshold.
bool all_below_or_equal(std::vector<float> values, float threshold);

/// Check if all values are equal or greater than threshold value.
/// \param values Vector of float values.
/// \param threshold Threshold value.
/// \return True if all values are equal or bigger than threshold.
bool all_over_or_equal(std::vector<float> values, float threshold);

/// Get id of contour which is most outer and biggest.
/// \param contours Vector of contours.
/// \param hierarchy Contour's hierarchies.
/// \return Best contour id.
long best_contour(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i>& hierarchy);

/// Get most possible out of all passed states, this was used with grayscale and HSV test.
/// \param states Pair of state and its value.
/// \return Most possible state.
TLState get_most_possible_state(std::vector<std::pair<TLState, float>> states);

/// Fixes possible error in mask by filling center hole with rectangle.
/// \param mask Mask to be fixed
void fix_mask(cv::Mat1b &mask);

/// Calculates average brigthness in image.
/// \param img OpenCV image.
/// \return Average brightness in image.
float get_average_brightness(cv::Mat & img);

/// Calculates mask coverage in image.
/// \param partInfo HSV test param.
/// \return Mask coverage in percents.
float get_mask_coverage(HsvTestParam * partInfo);

/// Starts HSV test, compatible to start in thread.
/// \param param Param should be HsvTestParam
void hsvTest(void *param);

/// Starts grayscale test, compatible to start in thread.
/// \param param Param should be GrayScaleTestParam
void grayScaleTest(void * param);

/// Find vertical boundaries (top and bottom) in image, first rows where non black pixels are present.
/// \param img OpenCV image.
/// \return Pair of (top, bottom) boundaries.
std::pair<int, int> find_vertical_boundaries(cv::Mat1b & img);

/// Sum pixels value for OpenCv Mat1b image type in given range.
/// \param img OpenCV image.
/// \param lowRow Lower row boundary.
/// \param highRow Upper row boundary.
/// \param lowCol Lower column boundary.
/// \param highCol Upper column boundary.
/// \param pixelCount Reference to pixel count.
/// \return Sum of pixels values
float get_sum_in_range(cv::Mat &img, int lowRow, int highRow, int lowCol, int highCol, int &pixelCount);

/// Check if background should be removed, by comparing border and center brightness.
/// \param grayImg Input OpenCV image.
/// \return True if border brightness is higher than center brightness.
bool should_remove_background(cv::Mat &grayImg);

/// Removes backgound for OpenCV images.
/// \param grayImg Reference to gray image.
/// \param hsvImg Reference to HSV image.
/// \param verbose True if more informations should be printed.
void remove_background(cv::Mat & grayImg, cv::Mat & hsvImg, bool verbose);

/// Check if HSV test is clear, one value is much bigger then others.
/// \param top Top mask coverage.
/// \param middle Middle mask coverage.
/// \param bottom Bottom mask coverage.
/// \return True if HSV test is clear.
bool clear_hsv_test(float top, float middle, float bottom);

/// Translate enum value to string.
/// \param state Traffic light state.
/// \return String representing state.
std::string translate_TL_state(TLState state);

/// Get color for traffic light state.
/// \param state Detected traffic light state.
/// \return Color for given traffic light state.
dlib::rgb_pixel get_color_for_state(TLState state);

/// Used to detect state with grayscale and HSV test. Now it is not used anymore.
/// \param img OpenCV image of detected traffic light.
/// \param verbose If more informations should be print.
/// \return Detected state on traffic light.
TLState get_traffic_light_state(cv::Mat & img, bool verbose = false);

/// Saves found object.
/// \param mat OpenCV image, where object was found.
/// \param rectangle Detection rectangle.
/// \param imgIndex Image index, used to construct file name.
/// \param labelIndex Label index, used to construct file name.
void save_found_crop(cv::Mat & mat, dlib::rectangle rectangle, int imgIndex, int labelIndex);

/// Saves found object, with possibility to rescale saved image (OpenCV).
/// \param mat OpenCV image, where object was found.
/// \param detRect Detection rectangle.
/// \param fileName file name of image.
/// \param sizeRect Rectangle used to rescale image.
void save_found_crop(cv::Mat & mat, dlib::rectangle detRect, std::string fileName, dlib::rectangle sizeRect = dlib::rectangle());

/// Saves found object, with possibility to rescale saved image (Dlib).
/// \param image Dlib image, where object was found.
/// \param detRect Detection rectangle.
/// \param fileName file name of image.
/// \param sizeRect Rectangle used to rescale image.
void save_found_crop(const dlib::matrix<dlib::rgb_pixel>& image, dlib::rectangle detRect, std::string fileName, dlib::rectangle sizeRect = dlib::rectangle());

/// Crops OpenCV image with dlib rectangle.
/// \param mat OpenCV image to crop.
/// \param cropRectangle Crop rectangle.
/// \return Cropped image.
cv::Mat crop_image(const cv::Mat & mat, const dlib::rectangle& cropRectangle);

/// Crop Dlib image.
/// \param original Original image.
/// \param cropRectangle Crop rectangle.
/// \param exactCrop [Obsolete]
/// \return Cropped image.
dlib::matrix<dlib::rgb_pixel>
crop_image(const dlib::matrix<dlib::rgb_pixel> &original, const dlib::rectangle& cropRectangle, bool exactCrop = false);

/// Check if rectangle is valid in image, inside its boundaries (OpenCV)
/// \param rect Rectangle to test.
/// \param img Image.
/// \return True if rectangle is valid.
bool valid_rectangle(const dlib::rectangle& rect, const cv::Mat& img);

/// Check if rectangle is valid in image, inside its boundaries (Dlib)
/// \param rect Rectangle to test.
/// \param img Image.
/// \return True if rectangle is valid.
bool valid_rectangle(const dlib::rectangle& rect, const dlib::matrix<dlib::rgb_pixel>& img);

/// Transform rectangle from scaled image back to original image space.
/// \param rect Scaled rectangle.
/// \param scaleFactor Scale factor used to scale image from original to scaled image size.
/// \return Rescaled rectangle to original image boundaries.
dlib::rectangle transform_rectangle_back(const dlib::rectangle& rect, const float scaleFactor);

/// Converts Dlib image to grayscale.
/// \param image Reference to Dlib image.
void convert_to_grayscale(dlib::matrix<dlib::rgb_pixel>& image);

/// Converts Dlib image to grayscale.
/// \param image Reference to OpenCV img.
void convert_to_grayscale(dlib::array2d<dlib::rgb_pixel>& image);

/// Get number of valid label boxes.
/// \param boxes All label boxes.
/// \return Number of valid label boxes.
int number_of_non_ignored_rectangles(const std::vector<dlib::mmod_rect> &boxes);

/// Check if given detection is correct.
/// \param detection Detected rectangle.
/// \param truthBoxes Grount truth rectangles for current image.
/// \return Pair of bools, (correct detection, correct state detection).
std::pair<bool, bool> is_correct_detection(const dlib::mmod_rect& detection, const std::vector<dlib::mmod_rect>& truthBoxes);

/// Get string annotation this rectangle.
/// \return Rectangle instance.
std::string to_str(const dlib::rectangle &);

#endif //DISPLAYIMAGE_OPENCVUTILS_H

