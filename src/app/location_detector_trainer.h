/** \file location_detector_trainer.h
 * Routines to train location detector.
 */

#ifndef TLD_TRAFFIC_LIGHT_DETECTOR_TRAINER_H
#define TLD_TRAFFIC_LIGHT_DETECTOR_TRAINER_H


#include "traffic_light_detector.h"

/// Template class for location detection of traffic lights.
/// \tparam LocationNetType Type of CNN.
template <
        typename LocationNetType
>
class location_detector_trainer {
private:

    /// Taken from http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html
    /// \param boxes Vector of bounding boxes.
    /// \param overlaps Function which determines if two boxes overlaps.
    /// \return Number of ignored boxes.
    int overlapped_boxes_count(std::vector<mmod_rect> boxes, const test_box_overlap &overlaps) {
        int num_ignored = 0;

        for (size_t i = 0; i < boxes.size(); ++i) {
            if (boxes[i].ignore)
                continue;

            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (boxes[j].ignore)
                    continue;

                if (overlaps(boxes[i], boxes[j])) {
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

    /// First training metod, training in mini batches.
    /// \param locationDatasetFile Xml file with data annotations.
    /// \param outFile Output file for serialization.
    void train_location_network(const std::string &locationDatasetFile, const std::string& outFile = "") {
        using namespace std;
        using namespace dlib;

        std::vector<matrix<rgb_pixel>> trainingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, locationDatasetFile);

        cout << "Loading and checking bounding boxes." << endl;

        int overlappingBoxesCount = 0;
        int tooSmallBoxesCount = 0;

        for (std::vector<mmod_rect> &boxes : trainingBoxes) {
            overlappingBoxesCount += overlapped_boxes_count(boxes, test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD));

            for (mmod_rect &boundingBox : boxes) {
                if (boundingBox.rect.width() < MIN_BOUNDING_BOX_SIZE &&
                    boundingBox.rect.height() < MIN_BOUNDING_BOX_SIZE) {
                    boundingBox.ignore = true;
                    ++tooSmallBoxesCount;
                }
            }
        }

        cout << "Number of boxes ignored because of overlapping: " << overlappingBoxesCount << endl;
        cout << "Number of boxes ignored because both sides are smaller than " << MIN_BOUNDING_BOX_SIZE << " : "
             << tooSmallBoxesCount << endl;


        cout << "Number of training images: " << trainingImages.size() << endl;

        mmod_options mmodOptions(trainingBoxes, DW_LONG_SIDE, DW_SHORT_SIDE);
        mmodOptions.overlaps_ignore = test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD);

        cout << "Number of detector windows " << mmodOptions.detector_windows.size() << endl;

        LocationNetType net(mmodOptions);

        net.subnet().layer_details().set_num_filters(mmodOptions.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<LocationNetType> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<LocationNetType> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif

        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);
        trainer.set_iterations_without_progress_threshold(TRAIN_ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_synchronization_file("TL_SYNC_FILE", std::chrono::minutes(SYNC_INTERVAL));

        random_cropper cropper;

        cropper.set_chip_dims(CHIP_HEIGHT, CHIP_WIDTH);
        cropper.set_min_object_size(MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);
        cropper.set_max_object_size(MAX_OBJECT_SIZE);
        cropper.set_background_crops_fraction(BACKGROUND_CROP_FRACTION);
        cropper.set_randomly_flip(RANDOM_FLIP);
        cropper.set_max_rotation_degrees(RANDOM_ROTATION_ANGLE);
        cropper.set_translate_amount(0);

        dlib::rand rnd;

        cout << trainer << cropper << endl;

        std::vector<matrix<rgb_pixel>> miniBatchImages;
        std::vector<std::vector<mmod_rect>> miniBatchLabels;


        while (trainer.get_learning_rate() >= MINIMAL_LEARNING_RATE) {
            cropper(BATCH_SIZE, trainingImages, trainingBoxes, miniBatchImages, miniBatchLabels);

            for (matrix<rgb_pixel> &img : miniBatchImages)
                disturb_colors(img, rnd);


            trainer.train_one_step(miniBatchImages, miniBatchLabels);
        }

        trainer.get_net();
        net.clean();

        std::string serializationFile = "TL_net.dat";
        if (outFile.length() != 0)
            serializationFile = outFile;

        serialize(serializationFile) << net;

        cout << "Training is completed." << endl;
        cout << "Training results: "
             << test_object_detection_function(net, trainingImages, trainingBoxes, test_box_overlap(), 0,
                                               mmodOptions.overlaps_ignore) << endl;
    }

    /*********************************************************************************************************************************************************/
    /// Second training method, with test every 30 training mini batches.
    /// \param locationDatasetFile Xml file with data annotations.
    /// \param testXmlFile Xml file with data annotations for test images.
    /// \param outFile Output file for serialization.
    void train_location_network_with_tests(const std::string &locationDatasetFile, const std::string& testXmlFile, const std::string& outFile = "")
    {
        using namespace std;
        using namespace dlib;

        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<matrix<rgb_pixel>>      testingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;
        std::vector<std::vector<mmod_rect>> testingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, locationDatasetFile);
        load_image_dataset(testingImages, testingBoxes, testXmlFile);

        cout << "Loading and checking bounding boxes." << endl;

        int num_overlapped_ignored_test = 0;
        for (std::vector<mmod_rect>& testBoxes : testingBoxes)
        {
            num_overlapped_ignored_test += overlapped_boxes_count(testBoxes, test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD));
        }

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

        cout << "Number of overlapped ignored testing bounding boxes: " << num_overlapped_ignored_test << endl;
        cout << "Number of boxes ignored because of overlapping: " << overlappingBoxesCount << endl;
        cout << "Number of boxes ignored because both sides are smaller than " << MIN_BOUNDING_BOX_SIZE << " : " << tooSmallBoxesCount << endl;
        cout << "Number of training images: " << trainingImages.size() << endl;
        cout << "Number of testing images: " << testingImages.size() << endl;

        mmod_options mmodOptions(trainingBoxes, MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);

        mmodOptions.overlaps_ignore = test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD);

        cout << "Number of detector windows " << mmodOptions.detector_windows.size() << endl;
        LocationNetType net(mmodOptions);

        net.subnet().layer_details().set_num_filters(mmodOptions.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<LocationNetType> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<LocationNetType> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif

        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);
        trainer.set_iterations_without_progress_threshold(ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_test_iterations_without_progress_threshold(TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_synchronization_file("TL_SYNC_FILE", std::chrono::minutes(SYNC_INTERVAL));

        random_cropper cropper;

        cropper.set_chip_dims(CHIP_HEIGHT, CHIP_WIDTH);
        cropper.set_min_object_size(MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);
        cropper.set_max_object_size(MAX_OBJECT_SIZE);
        cropper.set_background_crops_fraction(BACKGROUND_CROP_FRACTION);
        cropper.set_randomly_flip(RANDOM_FLIP);
        cropper.set_max_rotation_degrees(RANDOM_ROTATION_ANGLE);

        dlib::rand rnd;

        cout << trainer << cropper << endl;

        std::vector<matrix<rgb_pixel>>      miniBatchImages;
        std::vector<std::vector<mmod_rect>> miniBatchLabels;

        int cnt = 1;

        while (trainer.get_learning_rate() >= MINIMAL_LEARNING_RATE)
        {
            if (cnt % 30 != 0 || testingImages.size() == 0)
            {
                cropper(BATCH_SIZE, trainingImages, trainingBoxes, miniBatchImages, miniBatchLabels);

                for (matrix<rgb_pixel> &img : miniBatchImages)
                    disturb_colors(img, rnd);


                trainer.train_one_step(miniBatchImages, miniBatchLabels);
            }
            else
            {
                cropper(BATCH_SIZE, testingImages, testingBoxes, miniBatchImages, miniBatchLabels);

                for (matrix<rgb_pixel> &img : miniBatchImages)
                    disturb_colors(img, rnd);

                trainer.test_one_step(miniBatchImages, miniBatchLabels);

            }

            ++cnt;
        }

        trainer.get_net();
        net.clean();

        std::string serializationFile = "TL_net.dat";
        if (outFile.length() != 0)
            serializationFile = outFile;

        serialize(serializationFile) << net;

        cout << "Training is completed." << endl;
    }
    /*********************************************************************************************************************************************************/

    /// Train shape predictor, to enhance traffic light detection.
    /// \param serializedLocationNet Serialized network.
    /// \param xmlFile XML file with data annotations.
    template <typename LocationTestNetType, typename StateTestNetType>
    void train_shape_predictor(const std::string& serializedLocationNet, const std::string& xmlFile)
    {
        using namespace std;
        using namespace dlib;

        string shapePredictorDatasetFile = xmlFile + "sp.xml";

        cout << "Getting detection rectangles." << endl;
        traffic_light_detector<LocationTestNetType, StateTestNetType> tl_detector;
        //std::vector<std::vector<mmod_rect>> detections = get_detected_rectanges(serializedLocationNet, xmlFile);
        std::vector<std::vector<mmod_rect>> detections = tl_detector.get_detected_rectanges_for_sp(serializedLocationNet, xmlFile);

        image_dataset_metadata::dataset xmlDataset, shapePredictorDataset;

        cout << "Loading xml dataset." << endl;
        image_dataset_metadata::load_image_dataset_metadata(xmlDataset, xmlFile);

        cout << "Creating shape predictor dataset." << endl;
        shapePredictorDataset = make_bounding_box_regression_training_data(xmlDataset, detections);

        cout << "Saving shape predictor dataset.." << endl;
        image_dataset_metadata::save_image_dataset_metadata(shapePredictorDataset, shapePredictorDatasetFile);

        std::cout << "Loaded shape predictor settings: " << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "Oversampling: " << OVERSAMPLING_AMOUNT << std::endl;
        std::cout << "NU: " << NU << std::endl;
        std::cout << "Tree depth: " << TREE_DEPTH  << std::endl;
        std::cout << "Thread count: " << THREAD_COUNT  << std::endl;
        std::cout << "=============================" << std::endl;

        cout << "Training shape predictor." << endl;

        std::vector<matrix<rgb_pixel>> trainImages;
        std::vector<std::vector<full_object_detection>> trainDetections;

        load_image_dataset(trainImages, trainDetections, shapePredictorDatasetFile);

        shape_predictor_trainer trainer;

        trainer.set_oversampling_amount(OVERSAMPLING_AMOUNT); //how many times increase number of training data
        trainer.set_nu(NU);
        trainer.set_tree_depth(TREE_DEPTH);

        trainer.set_num_threads(THREAD_COUNT);

        trainer.be_verbose();

        shape_predictor sp = trainer.train(trainImages, trainDetections);

        cout << "Mean training error: " << test_shape_predictor(sp, trainImages, trainDetections) << endl;

        LocationNetType net;
        cout << "Deserializing net." << endl;
        deserialize(serializedLocationNet) >> net;

        cout << "Serializing net and shape predictor ." << endl;
        serialize(serializedLocationNet) << net << sp;
    }
};

#endif //TLD_TRAFFIC_LIGHT_DETECTOR_TRAINER_H