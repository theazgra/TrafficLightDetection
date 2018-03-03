//
// Created by azgra on 17.1.18.
//
#include "settings.h"

std::vector<int> CUDA_DEVICES = std::vector<int>();
uint BATCH_SIZE = 20;
uint CHIP_WIDTH = 200;
uint CHIP_HEIGHT = 500;
uint MIN_OBJECT_SIZE_S = 13;
uint MIN_OBJECT_SIZE_L = 25;
uint RANDOM_ROTATION_ANGLE = 2;
float MAX_OBJECT_SIZE = 0.5f;
float BACKGROUND_CROP_FRACTION = 0.25f;
uint MIN_BOUNDING_BOX_SIZE = 20;
float OVERLAP_IOU = 0.5f;
float COVERED_THRESHOLD = 0.9f;
uint DW_LONG_SIDE = 25;
uint DW_SHORT_SIDE = 13;
uint SYNC_INTERVAL = 5;
float SGD_WEIGHT_DECAY = 0.0005f;
float SGD_MOMENTUM = 0.9f;
float LEARNING_RATE = 0.1f;
float MINIMAL_LEARNING_RATE = 0.0001f;
uint TRAINING_METHOD = 1;
int TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD = 500;
int ITERATION_WITHOUT_PROGRESS_THRESHOLD = 50000;
int TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD = 1000;
unsigned long OVERSAMPLING_AMOUNT = 5;
double NU = 0.05;
unsigned long TREE_DEPTH = 2;
unsigned long THREAD_COUNT = 10;

using namespace pugi;

bool load_settings(const char* xmlSettingsFile)
{
    xml_document document;
    xml_parse_result result = document.load_file(xmlSettingsFile);

    std::cout << result.status << std::endl;
    if (result.status != status_ok)
    {
        std::cout << "Error while parsing xml file containing settings." << std::endl;
        std::cout << result.description() << std::endl;
        return false;
    }

    xml_node root = document.child("app_settings");

    //Cuda settings
    xml_node cuda_devices = root.child("cuda_devices");
    for (xml_node cuda_device : cuda_devices.children("device"))
    {
        CUDA_DEVICES.push_back(cuda_device.text().as_int());
        std::cout << "Setting cuda device: " << cuda_device.text().as_int() << std::endl;
    }

    //Cropper settings
    xml_node cropper = root.child("cropper");

    BATCH_SIZE = cropper.child("batch_size").text().as_uint(20);
    CHIP_WIDTH = cropper.child("chip_width").text().as_uint(200);
    CHIP_HEIGHT = cropper.child("chip_height").text().as_uint(500);
    MIN_OBJECT_SIZE_S = cropper.child("min_object_size_smaller_side").text().as_uint(13);
    MIN_OBJECT_SIZE_L = cropper.child("min_object_size_larger_side").text().as_uint(25);
    RANDOM_ROTATION_ANGLE = cropper.child("random_rotation_angle").text().as_uint(2);
    MAX_OBJECT_SIZE = cropper.child("max_object_size").text().as_float(0.5f);
    BACKGROUND_CROP_FRACTION = cropper.child("background_crop_fraction").text().as_float(0.25f);

    std::cout << "Loaded cropper settings: " << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "Chip dimensions: " << CHIP_WIDTH << " x " << CHIP_HEIGHT << std::endl;
    std::cout << "Minimal object size: Small size: " << MIN_OBJECT_SIZE_S << " ,larger size: " << MIN_OBJECT_SIZE_L << std::endl;
    std::cout << "Random rotation angle: " << RANDOM_ROTATION_ANGLE << ", max object size: " <<  MAX_OBJECT_SIZE << std::endl;
    std::cout << "Backgroud crop fraction: " << BACKGROUND_CROP_FRACTION << std::endl;
    std::cout << "=============================" << std::endl;
    //mmod settings
    xml_node mmod = root.child("mmod_settings");

    MIN_BOUNDING_BOX_SIZE = mmod.child("min_bounding_box_size").text().as_uint(20);
    OVERLAP_IOU = mmod.child("overlap_iou").text().as_float(0.5);
    COVERED_THRESHOLD = mmod.child("covered_threshold").text().as_float(0.9);
    DW_LONG_SIDE = mmod.child("detector_window").child("long_side").text().as_uint(25);
    DW_SHORT_SIDE = mmod.child("detector_window").child("short_side").text().as_uint(13);

    //Sync settings
    SYNC_INTERVAL = root.child("sync_settings").child("sync_interval").text().as_uint(5);

    //net settings
    xml_node net = root.child("net_settings");

    SGD_WEIGHT_DECAY = net.child("sgd_weight_decay").text().as_float(0.0005);
    SGD_MOMENTUM = net.child("sgd_momentum").text().as_float(0.9);
    LEARNING_RATE = net.child("learning_rate").text().as_float(0.1);
    MINIMAL_LEARNING_RATE = net.child("minimal_learning_rate").text().as_float(0.0001);

    //training settings
    xml_node training = net.child("training");
    TRAINING_METHOD = training.child("method").text().as_uint(1);

    TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD =
            training.child("tr_iter_wo_prog_threshold").text().as_int(500);

    ITERATION_WITHOUT_PROGRESS_THRESHOLD =
            training.child("iter_wo_prog_threshold").text().as_int(50000);

    TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD =
            training.child("test_iter_wo_prog_threshold").text().as_int(1000);

    xml_node spTrainer = root.child("shape_predictor_trainer");

    OVERSAMPLING_AMOUNT = spTrainer.child("oversampling_amount").text().as_ullong(5);
    NU = spTrainer.child("nu_value").text().as_double(0.05);
    TREE_DEPTH = spTrainer.child("tree_depth").text().as_ullong(2);
    THREAD_COUNT = spTrainer.child("thread_count").text().as_ullong(10);

    return true;
}

