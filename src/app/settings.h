
#ifndef SETTINGS_H
#define SETTINGS_H

#include "net_definition.h"
#include <vector>


//#define MULTIPLE_GPUS
const std::vector<int> CUDA_DEVICES = {0, 1, 2, 3}; //{0, 1, 2, 3};

///*******<BOUNDING BOX SETTINGS>*******

/**
 * Minimal size of bounding box.
 * If both width and height are smaller than this constant, bounding box is ignored.
 */
const unsigned int MIN_BOUNDING_BOX_SIZE = 20;

/**
 * If intersect of two boxes over their union is larget than this value, they are overlapping.
 */
const float OVERLAP_IOU = 0.5;

/**
 * If box is inside another box, and percentage covered is higher than this value then the covered box is ignored.
 */
const float COVERED_THRESHOLD = 0.9;

/**
 * Determines the longest side of the detector window in pixels. Unless shortest side is smalled than MIN_TARGET_SIZE,
 * in that case the shortest side will be set to MIN_TARGET_SIZE and the longest side will be set automatically to preserve
 * aspect ratio of the detector window.
 */
///How much pixels must the boundix box have in its longer side to still be recognizible
//original 25x13, testing new values
const int DW_LONG_SIDE = 25;
///How much pixels must the boundix box have in its shorter side to still be recognizible
const int DW_SHORT_SIDE = 13;
///*******</BOUNDING BOX SETTINGS>*******

///*******<CROPPER SETTINGS>*******
/**
 * Size of cropped images.
 */
const unsigned int CHIP_WIDTH = 200;//200;//400;
const unsigned int CHIP_HEIGHT = 500;//500;//1000;

/**
 * Angle used for random rotation.
 */
const unsigned int RANDOM_ROTATION_ANGLE = 2;

/**
 * Minimal object sizes
 */
///short side
const long MIN_OBJECT_SIZE_S = 13;
///long side
const long MIN_OBJECT_SIZE_L = 25;

/**
 * Maximal object size in relation to CHIP_SIZE
 */
const float MAX_OBJECT_SIZE = 0.5f;

/**
 * True if random flip images.
 */
const bool RANDOM_FLIP = false;
/**
* Defines fraction of crops with negative targets
*/
const float BACKGROUND_CROP_FRACTION = 0.25f;
/**
 * Batch size for one step.
 * If we set high batch size with small dataset we will overfit our network model.
 */
const unsigned int BATCH_SIZE = 20;//30;

///*******</CROPPER SETTINGS>*******

///*******<SYNC SETTINGS>*******

/**
 * Synchronization interval in minutes.
 */
const unsigned int SYNC_INTERVAL = 5;
///*******</SYNC SETTINGS>*******

///*******<NET SETTINGS>*******
const float SGD_WEIGHT_DECAY = 0.0005;
const float SGD_MOMENTUM = 0.9;
const float LEARNING_RATE = 0.1;
const float TARGET_LEARNING_RATE = 0.0001; //1e-4
/**
 * When testing or training will not progress in given iteration count, learning rate will be lowered.
 */
///Settings for first method pure learning without testing
const int TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD = 500;
///Settings for second method where every 30 iteration we will the test network
const int ITERATION_WITHOUT_PROGRESS_THRESHOLD = 50000;
const int TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD = 1000;

///*******</NET SETTINGS>*******



#endif //SETTINGS_H
