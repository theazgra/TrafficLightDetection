#ifndef BACHELOR_CAR_DETECTOR_H
#define BACHELOR_CAR_DETECTOR_H

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

void car_detection(char* folder, bool train, bool test);

#endif //BACHELOR_CAR_DETECTOR_H
