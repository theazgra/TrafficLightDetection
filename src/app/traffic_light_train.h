#ifndef BACHELOR_TRAFFIC_LIGHT_TRAIN_H
#define BACHELOR_TRAFFIC_LIGHT_TRAIN_H

#include "traffic_light_test.h"
#include <dlib/svm_threaded.h>

int overlapped_boxes_count(std::vector<dlib::mmod_rect> boxes, const dlib::test_box_overlap& overlaps);

void train(const std::string trainFile);
void train(const std::string trainFile, const std::string testFile);

void train_state(const std::string trainFile);


void train_shape_predictor(const std::string netFile, const std::string xmlFile);
void train_shape_predictor_for_state(const std::string netFile, const std::string xmlFile);



#endif //BACHELOR_TRAFFIC_LIGHT_TRAIN_H
