/** \file settings.h
 * All global variables are defined in this file.
 * Those variables are then loaded from xml file.
 */

#ifndef SETTINGS_H
#define SETTINGS_H

/// Define use of more CUDA devices, set in xml settings file.
#define MULTIPLE_GPUS

#include <vector>
#include "extern_files/pugixml.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include "net_definition.h"
#include "cv_utils.h"
#include "Stopwatch.h"

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

enum CudaJobType
{
    SaveCrops,
    SaveImages
};

/// This passes importatnt informations about job to be executed on cuda device.
struct CudaJobInfo
{
    std::string netFile;
    std::string stateNetFile;
    std::string resultFolder;
    std::vector<dlib::matrix<dlib::rgb_pixel>> jobImages;
    Stopwatch stopwatch;
    int deviceId;
    ulong frameIndexOffset;
    ulong begin;
    ulong end;
    CudaJobType jobType;

    dlib::rectangle sizeRectangle;

    CudaJobInfo(
            int deviceId,
            const std::string& netFile,
            const std::string& stateNetFile,
            const std::string& resultFolder,
            const std::vector<dlib::matrix<dlib::rgb_pixel>>& jobImages,
            ulong begin,
            ulong end,
            CudaJobType jobType)
    {
        this->netFile = netFile;
        this->stateNetFile = stateNetFile;
        this->deviceId = deviceId;
        this->begin = begin;
        this->end = end;
        this->jobImages = jobImages;

        this->stopwatch = Stopwatch("Cuda device: " + std::to_string(deviceId));
        this->frameIndexOffset = begin;
        this->resultFolder = resultFolder;

        this->jobType = jobType;
    }
};

/// Loads settings from xml file. Setting global variables.
/// \param xmlSettingsFile XML file with settings.
/// \return True if load was successfull.
bool load_settings(const char* xmlSettingsFile);

/// Scale factor of image. Image is resized by this before being pushed into test network.
extern float FRAME_SCALING;

///If only some part of image should be tested (pushed to test network) instead of whole image.
extern bool ONLY_TOP_HALF;

/// Set cuda devices.
extern std::vector<int> CUDA_DEVICES;// = {0, 1, 2, 3}; //{0, 1, 2, 3};

///*******<BOUNDING BOX SETTINGS>*******

/// Minimal size of bounding box. If both width and height are smaller than this constant, bounding box is ignored.
extern unsigned int MIN_BOUNDING_BOX_SIZE;

/// If intersect of two boxes over their union is larget than this value, they are overlapping.
extern float OVERLAP_IOU;

///If box is inside another box, and percentage covered is higher than this value then the covered box is ignored.
extern float COVERED_THRESHOLD;

/// How much pixels must the boundix box have in its longer side to still be recognizible
extern unsigned int DW_LONG_SIDE;
/// How much pixels must the boundix box have in its shorter side to still be recognizible
extern unsigned int DW_SHORT_SIDE;
///*******</BOUNDING BOX SETTINGS>*******

///*******<CROPPER SETTINGS>*******
/// Width of cropped images.
extern unsigned int CHIP_WIDTH;//200;//400;
/// Height of cropped images.
extern unsigned int CHIP_HEIGHT;//500;//1000;

/// Angle used for random rotation.
extern unsigned int RANDOM_ROTATION_ANGLE;

/// Minimal object sizes (short side)
extern unsigned int MIN_OBJECT_SIZE_S;
/// Minimal object sizes (long side)
extern unsigned int MIN_OBJECT_SIZE_L;

/// Maximal object size in relation to CHIP_SIZE
extern float MAX_OBJECT_SIZE;

/// True if random flip images.
const bool RANDOM_FLIP = false;

/// Defines fraction of crops with negative targets
extern float BACKGROUND_CROP_FRACTION;

/// Batch size for one step. If we set high batch size with small dataset we will overfit our network model.
extern unsigned int BATCH_SIZE;//30;

///*******</CROPPER SETTINGS>*******

///*******<SYNC SETTINGS>*******

/// Synchronization interval in minutes.
extern unsigned int SYNC_INTERVAL;
///*******</SYNC SETTINGS>*******

///*******<NET SETTINGS>*******

/// Weight decay of standart gradien descent.
extern float SGD_WEIGHT_DECAY;
/// Momentum of standart gradien descent.
extern float SGD_MOMENTUM;
/// Learning rate of network.
extern float LEARNING_RATE;

/// Minimal learning rate.
extern float MINIMAL_LEARNING_RATE; //1e-4

/// Training method id.
extern unsigned int TRAINING_METHOD;

/**
 * When testing or training will not progress in given iteration count, learning rate will be lowered.
 */
/// TRAIN ITERATION WITHOUT PROGRESS THRESHOLD for (method 1)
extern int TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD;
/// TRAIN ITERATION WITHOUT PROGRESS THRESHOLD for (method 2)
extern int ITERATION_WITHOUT_PROGRESS_THRESHOLD;

/// TEST ITERATION WITHOUT PROGRESS THRESHOLD for (method 1)
extern int TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD;

///*******</NET SETTINGS>*******

///*******<SHAPE PREDICTOR TRAINER SETTIGNS>*******

/// How many times increase number of training data
extern unsigned long OVERSAMPLING_AMOUNT;

/// Larger values better fitting but may cause overfitting.
extern double NU;

/// Depth of the trees used in the cascade.
extern unsigned long TREE_DEPTH;

/// Thread count.
extern unsigned long THREAD_COUNT;

/// State net mmod window bigger side.
extern unsigned long STATE_WINDOW_WIDTH;
/// State net mmod window smaller side.
extern unsigned long STATE_WINDOW_HEIGHT;
/// State net chip width.
extern unsigned long STATE_CHIP_WIDTH;
/// State net chip height.
extern unsigned long STATE_CHIP_HEIGHT;
/// State net batch size.
extern unsigned long STATE_BATCH_SIZE;
/// State net train iterations without progress.
extern unsigned long STATE_ITERATION_WITHOUT_PROGRESS_THRESHOLD;

/// Crop saving (Crop width).
extern unsigned int CROP_WIDTH;
/// Crop saving (Crop height)
extern unsigned int CROP_HEIGHT;

#endif //SETTINGS_H
