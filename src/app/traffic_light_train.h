/** \file traffic_light_train.h
 * Routines moved to location_detector_trainer.h and state_detector_trainer.h
 */

#ifndef BACHELOR_TRAFFIC_LIGHT_TRAIN_H
#define BACHELOR_TRAFFIC_LIGHT_TRAIN_H

#include "traffic_light_test.h"
#include <dlib/svm_threaded.h>

/// Taken from http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html
/// \param boxes Vector of bounding boxes.
/// \param overlaps Function which determines if two boxes overlaps.
/// \return Number of ignored boxes.
int overlapped_boxes_count(std::vector<dlib::mmod_rect> boxes, const dlib::test_box_overlap& overlaps);


void train(const std::string trainFile);

/// Second training method (train-test method)
/// \param trainFile XML file with data annotations.
/// \param testFile XML file with data annotations of test images.
void train(const std::string trainFile, const std::string testFile);

/// Trains resnet.
/// \param trainFile XML file with data annotations.
void train_resnet(const std::string trainFile);

/// Train network for detection of state.
/// \param trainFile XML file with data annotations.
/// \param outFile Serialization file.
void train_state(const std::string trainFile, const std::string outFile = "");

/// Train shape predictor, to enhance traffic light detection.
/// \param netFile Serialized network.
/// \param xmlFile XML file with data annotations.
void train_shape_predictor(const std::string netFile, const std::string xmlFile);

#endif //BACHELOR_TRAFFIC_LIGHT_TRAIN_H
