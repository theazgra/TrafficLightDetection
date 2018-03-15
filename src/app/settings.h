#ifndef SETTINGS_H
#define SETTINGS_H

#include "net_definition.h"
#include <vector>
#include "pugi_xml/pugixml.hpp"

bool load_settings(const char* xmlSettingsFile);


#define MULTIPLE_GPUS
extern std::vector<int> CUDA_DEVICES;// = {0, 1, 2, 3}; //{0, 1, 2, 3};

///*******<BOUNDING BOX SETTINGS>*******

/**
 * Minimal size of bounding box.
 * If both width and height are smaller than this constant, bounding box is ignored.
 */
extern unsigned int MIN_BOUNDING_BOX_SIZE;

/**
 * If intersect of two boxes over their union is larget than this value, they are overlapping.
 */
extern float OVERLAP_IOU;

/**
 * If box is inside another box, and percentage covered is higher than this value then the covered box is ignored.
 */
extern float COVERED_THRESHOLD;

/**
 * Determines the longest side of the detector window in pixels. Unless shortest side is smalled than MIN_TARGET_SIZE,
 * in that case the shortest side will be set to MIN_TARGET_SIZE and the longest side will be set automatically to preserve
 * aspect ratio of the detector window.
 */
///How much pixels must the boundix box have in its longer side to still be recognizible
//original 25x13, testing new values
extern unsigned int DW_LONG_SIDE;
///How much pixels must the boundix box have in its shorter side to still be recognizible
extern unsigned int DW_SHORT_SIDE;
///*******</BOUNDING BOX SETTINGS>*******

///*******<CROPPER SETTINGS>*******
/**
 * Size of cropped images.
 */
extern unsigned int CHIP_WIDTH;//200;//400;
extern unsigned int CHIP_HEIGHT;//500;//1000;

/**
 * Angle used for random rotation.
 */
extern unsigned int RANDOM_ROTATION_ANGLE;

/**
 * Minimal object sizes
 */
///short side
extern unsigned int MIN_OBJECT_SIZE_S;
///long side
extern unsigned int MIN_OBJECT_SIZE_L;

/**
 * Maximal object size in relation to CHIP_SIZE
 */
extern float MAX_OBJECT_SIZE;

/**
 * True if random flip images.
 */
const bool RANDOM_FLIP = false;
/**
* Defines fraction of crops with negative targets
*/
extern float BACKGROUND_CROP_FRACTION;
/**
 * Batch size for one step.
 * If we set high batch size with small dataset we will overfit our network model.
 */
extern unsigned int BATCH_SIZE;//30;

///*******</CROPPER SETTINGS>*******

///*******<SYNC SETTINGS>*******

/**
 * Synchronization interval in minutes.
 */
extern unsigned int SYNC_INTERVAL;
///*******</SYNC SETTINGS>*******

///*******<NET SETTINGS>*******
extern float SGD_WEIGHT_DECAY;
extern float SGD_MOMENTUM;
extern float LEARNING_RATE;
extern float MINIMAL_LEARNING_RATE; //1e-4

extern unsigned int TRAINING_METHOD;

/**
 * When testing or training will not progress in given iteration count, learning rate will be lowered.
 */
///Settings for first method pure learning without testing
extern int TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD;
///Settings for second method where every 30 iteration we will the test network
extern int ITERATION_WITHOUT_PROGRESS_THRESHOLD;
extern int TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD;

///*******</NET SETTINGS>*******

///*******<SHAPE PREDICTOR TRAINER SETTIGNS>*******

/**
 * How many times increase number of training data
 */
extern unsigned long OVERSAMPLING_AMOUNT;

/**
 * Larger values better fitting but may cause overfitting.
 */
extern double NU;
/**
 * Depth of the trees used in the cascade.
 */
extern unsigned long TREE_DEPTH;
/**
 * Thread count.
 */
extern unsigned long THREAD_COUNT;

///*******<FHOG DETECTOR SETTIGNS>*******
extern unsigned long FHOG_WINDOW_WIDTH;
extern unsigned long FHOG_WINDOW_HEIGHT;
extern unsigned long FHOG_THREAD_COUNT;
extern float FHOG_C;
extern float FHOG_EPSILON;

extern unsigned int CROP_WIDTH;
extern unsigned int CROP_HEIGHT;

#endif //SETTINGS_H
