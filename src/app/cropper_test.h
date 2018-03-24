/** \file cropper_test.h
 * Functions to test random cropper with set parameters.
 */
#ifndef BACHELOR_CROPPER_TEST_H
#define BACHELOR_CROPPER_TEST_H

#include "settings.h"
#include <iostream>
#include "traffic_light_train.h"
#include <dlib/image_transforms.h>

/// Test random cropper with parameters set in app_settings.xml
/// \param xmlFile XML file containing data annotations.
/// \param display If window displaying crops should be shown.
/// \param displayOnly If only window displaying crops should be shown, skip checking function.
void test_cropper(const std::string xmlFile, bool display, bool displayOnly);

#endif //BACHELOR_CROPPER_TEST_H
