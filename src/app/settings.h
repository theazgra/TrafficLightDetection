
#ifndef SETTINGS_H
#define SETTINGS_H

//#define MULTIPLE_GPUS

///*******<BOUNDING BOX SETTINGS>*******

/**
 * Minimal size of bounding box.
 * If both width and height are smaller than this constant, bounding box is ignored.
 */
const unsigned int MIN_BOUNDING_BOX_SIZE = 35;

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
const int TARGET_SIZE = 60;
const int MIN_TARGET_SIZE = 20;
///*******</BOUNDING BOX SETTINGS>*******

///*******<CROPPER SETTINGS>*******
/**
 * Size of cropped images.
 */
const unsigned int CHIP_SIZE = 200;

const unsigned int CHIP_WIDTH = 400;
const unsigned int CHIP_HEIGHT = 1000;

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
 * Batch size for one step.
 */
const unsigned int BATCH_SIZE = 100;

///*******</CROPPER SETTINGS>*******

///*******<SYNC SETTINGS>*******
/**
 * Synchronization file.
 */
//const char* NET_SYNC_FILE = "TL_network_sync";

/**
 * Synchronization interval in minutes.
 */
const unsigned int SYNC_INTERVAL = 5;
///*******</SYNC SETTINGS>*******

///*******<NET SETTINGS>*******
const float SGD_WEIGHT_DECAY = 0.0005;
const float SGD_MOMENTUM = 0.9;
const float LEARNING_RATE = 0.1;
const float LEARNING_RATE_LIMIT = 0.0001; //1e-4
/**
 * When testing or training will not progress in given iteration count, learning rate will be lowered.
 * This settings means that training progress is not that important as testing progress. Once we observe TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD
 * testing iterations without progress we will lower the learning rate.
 */
const int ITERATION_WITHOUT_PROGRESS_THRESHOLD = 50000;
const int TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD = 1000;

//const char* NET_DAT_FILE = "TL_network.dat";

///*******</NET SETTINGS>*******



#endif //SETTINGS_H
