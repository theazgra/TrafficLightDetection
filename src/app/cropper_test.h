#ifndef BACHELOR_CROPPER_TEST_H
#define BACHELOR_CROPPER_TEST_H


const unsigned int CHIP_SIZE = 200;
const unsigned int RANDOM_ROTATION_ANGLE = 2;
const float MIN_SIZE = 0.4f;
const float MAX_SIZE = 0.9f;
const bool RANDOM_FLIP = true;
const unsigned int BATCH_SIZE = 30;

void test_cropper(char* argv);

#endif //BACHELOR_CROPPER_TEST_H
