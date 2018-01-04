#ifndef BACHELOR_TRAFFIC_LIGHT_TRAIN_H
#define BACHELOR_TRAFFIC_LIGHT_TRAIN_H

#include "settings.h"

void train(const std::string trainFile);
void train(const std::string trainFile, const std::string testFile);
int overlapped_boxes_count(std::vector<dlib::mmod_rect> boxes, const dlib::test_box_overlap& overlaps);

#endif //BACHELOR_TRAFFIC_LIGHT_TRAIN_H
