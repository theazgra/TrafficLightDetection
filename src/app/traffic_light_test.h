/** \file traffic_light_test.h
 * Testing methods.
 */

#ifndef BACHELOR_TRAFFIC_LIGHT_TEST_H
#define BACHELOR_TRAFFIC_LIGHT_TEST_H

#include <iostream>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include "settings.h"
#include "Stopwatch.h"
#include "cv_utils.h"

/// Test type
enum TestType
{
    /// Dipslay images
    Display,
    /// Don't display images.
    NoDisplay,
    /// Display only images with errors in detection
    OnlyErrorDisplay
};

/// Basic test method.
/// \param netFile Serialized network.
/// \param testFile Xml file with data annotations.
/// \param testType Test type.
void test(std::string netFile, std::string testFile, TestType testType);

/// Save detected objects.
/// \param netFile Serialized network.
/// \param xmlFile Xml file with data annotations.
/// \param folderPath Where to save images.
/// \param sizeRect Optional image size.
void save_detected_objects(const std::string netFile, const std::string xmlFile, const std::string folderPath, dlib::rectangle sizeRect = dlib::rectangle());

/// Save processed frames from video file.
/// \param netFile Serialized network.
/// \param videoFile Video file.
/// \param resultFolder Where to save images.
void save_video(std::string netFile, std::string videoFile, std::string resultFolder);

/// Save processed from from xml file.
/// \param netFile Serialized network.
/// \param xmlFile Xml file with data annotations.
/// \param resultFolder Where to save images.
void save_video_frames(std::string netFile, std::string xmlFile, std::string resultFolder);

/// Save processed from from xml file, uses shape predictor to enhance prediction.
/// \param netFile Serialized network.
/// \param xmlFile Xml file with data annotations.
/// \param resultFolder Where to save images.
void save_video_frames_with_sp(std::string netFile, std::string xmlFile, std::string resultFolder);

/// Save processed from from xml file, uses shape predictor to enhance prediction and state net to detect traffic light state.
/// \param netFile Serialized network.
/// \param stateNetFile
/// \param xmlFile Xml file with data annotations.
/// \param resultFolder Where to save images.
void save_video_frames_with_sp2(const std::string netFile, const std::string stateNetFile,
                                const std::string xmlFile, const std::string resultFolder);

/// Save processed from from xml file, uses shape predictor to enhance prediction and state net to detect traffic light state. (ResNet)
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param xmlFile Xml file with data annotations.
/// \param resultFolder Where to save images.
void resnet_save_video_frames_with_sp2( const std::string netFile, const std::string stateNetFile,
                                        const std::string xmlFile, const std::string resultFolder);

/// Save visualization images from CNN.
/// \param netFile Serialized net file.
/// \param imgFile Image file.
void visualize_detection(std::string netFile, std::string imgFile);

/// Get detected object locations.
/// \param netFile Serialized network.
/// \param xmlFile Xml file with data annotations.
/// \return Vector of detected objects locations.
std::vector<std::vector<dlib::mmod_rect>> get_detected_rectanges(const std::string netFile, const std::string xmlFile);

/// Get number of valid label boxes.
/// \param boxes All label boxes.
/// \return Number of valid label boxes.
int number_of_label_boxes(std::vector<dlib::mmod_rect> boxes);

/// Detect state of traffic light.
/// \param netFile Serialized state network.
/// \param dlibImg Image of found traffic light.
/// \return Detected state.
TLState detect_state(const std::string netFile, const dlib::matrix<dlib::rgb_pixel> dlibImg);

/// Get detected state based, based on state net detections.
/// \param detections State net detections.
/// \param image Found traffic light image.
/// \return Detected state.
TLState get_detected_state(const std::vector<dlib::mmod_rect>& detections, const dlib::matrix<dlib::rgb_pixel>& image);
#endif //BACHELOR_TRAFFIC_LIGHT_TEST_H