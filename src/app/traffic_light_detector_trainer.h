#ifndef TLD_TRAFFIC_LIGHT_DETECTOR_TRAINER_H
#define TLD_TRAFFIC_LIGHT_DETECTOR_TRAINER_H


#include <iostream>
#include <string>
#include "settings.h"


template <
        typename LocationNetType,
        typename StateNetType
>
class Traffic_light_detector_trainer
{
private:
    LocationNetType locationTrainNet;
    StateNetType stateTrainNet;

    std::string locationDatasetFile;
    std::string stateDatasetFile;

    int overlapped_boxes_count(std::vector<mmod_rect> boxes, const test_box_overlap& overlaps)
    {
        int num_ignored = 0;

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (boxes[i].ignore)
                continue;

            for (size_t j = i + 1; j < boxes.size(); ++j)
            {
                if (boxes[j].ignore)
                    continue;

                if (overlaps(boxes[i], boxes[j]))
                {
                    ++num_ignored;
                    if (boxes[i].rect.area() < boxes[j].rect.area())
                        boxes[i].ignore = true;
                    else
                        boxes[j].ignore = true;

                }
            }
        }

        return num_ignored;
    }

public:

    Traffic_light_detector_trainer(const std::string& locationDataset, const std::string& stateDataset)
    {
        this->locationDatasetFile = locationDataset;
        this->stateDatasetFile = stateDataset;
    }

    void train_location_network()
    {
        using namespace std;
        using namespace dlib;

        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, this->locationDatasetFile);

        cout << "Loading and checking bounding boxes." << endl;

        int overlappingBoxesCount = 0;
        int tooSmallBoxesCount = 0;

        for (std::vector<mmod_rect>& boxes : trainingBoxes)
        {
            overlappingBoxesCount += overlapped_boxes_count(boxes, test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD));

            for (mmod_rect& boundingBox : boxes)
            {
                if (boundingBox.rect.width() < MIN_BOUNDING_BOX_SIZE && boundingBox.rect.height() < MIN_BOUNDING_BOX_SIZE)
                {
                    boundingBox.ignore = true;
                    ++tooSmallBoxesCount;
                }
            }
        }

        cout << "Number of boxes ignored because of overlapping: " << overlappingBoxesCount << endl;
        cout << "Number of boxes ignored because both sides are smaller than " << MIN_BOUNDING_BOX_SIZE << " : " << tooSmallBoxesCount << endl;


        cout << "Number of training images: " << trainingImages.size() << endl;

        mmod_options options(trainingBoxes, DW_LONG_SIDE, DW_SHORT_SIDE);
        options.overlaps_ignore = test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD);

        cout << "Number of detector windows " << options.detector_windows.size() << endl;

        LocationNetType net(options);

        net.subnet().layer_details().set_num_filters(options.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif

        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);
        trainer.set_iterations_without_progress_threshold(TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_synchronization_file("TL_SYNC_FILE", std::chrono::minutes(SYNC_INTERVAL));

        random_cropper cropper;

        //rows then cols
        cropper.set_chip_dims(CHIP_HEIGHT, CHIP_WIDTH);
        cropper.set_min_object_size(MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);
        cropper.set_max_object_size(MAX_OBJECT_SIZE);

        cropper.set_background_crops_fraction(BACKGROUND_CROP_FRACTION);

        //defaulted to false
        cropper.set_randomly_flip(RANDOM_FLIP);
        cropper.set_max_rotation_degrees(RANDOM_ROTATION_ANGLE);
        cropper.set_translate_amount(0);

        dlib::rand rnd;

        cout << trainer << cropper << endl;

        std::vector<matrix<rgb_pixel>>      miniBatchImages;
        std::vector<std::vector<mmod_rect>> miniBatchLabels;


        while (trainer.get_learning_rate() >= MINIMAL_LEARNING_RATE)
        {
            cropper(BATCH_SIZE, trainingImages, trainingBoxes, miniBatchImages, miniBatchLabels);

            for (matrix<rgb_pixel> &img : miniBatchImages)
                disturb_colors(img, rnd);


            trainer.train_one_step(miniBatchImages, miniBatchLabels);
        }

        trainer.get_net();
        net.clean();
        serialize("TL_net.dat") << net;

        cout << "Training is completed." << endl;
        cout << "Training results: " << test_object_detection_function(net, trainingImages, trainingBoxes, test_box_overlap(), 0, options.overlaps_ignore) << endl;
    }
};

#endif //TLD_TRAFFIC_LIGHT_DETECTOR_TRAINER_H