#include "traffic_light_train.h"
#include "Logger.h"

//#define WINDOW

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
/*********************************************************************************************************************************************************/
void train(const std::string trainFile)
{
    using namespace std;
    using namespace dlib;
    //"Debug" mode.
    try
    {
        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, trainFile);

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
        net_type net(options);

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
    catch (std::exception& e)
    {
        cout << "*****EXCEPTION*****" << endl;
        cout << e.what() << endl;
        cout << "*******************" << endl;
    }
}
/*********************************************************************************************************************************************************/
void train(const std::string trainFile, const std::string testFile)
{
    using namespace std;
    using namespace dlib;
    //"Debug" mode.
    try
    {
        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<matrix<rgb_pixel>>      testingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;
        std::vector<std::vector<mmod_rect>> testingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, trainFile);
        load_image_dataset(testingImages, testingBoxes, testFile);

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

        //mmod_options options(trainingBoxes, MIN_LONG_SIDE_SIZE, MIN_SMALL_SIDE_SIZE);
        mmod_options options(trainingBoxes, MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);

        options.overlaps_ignore = test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD);

        cout << "Number of detector windows " << options.detector_windows.size() << endl;
        net_type net(options);

        net.subnet().layer_details().set_num_filters(options.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif

        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);

        trainer.set_iterations_without_progress_threshold(ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_test_iterations_without_progress_threshold(TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD);


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
        serialize("TL_net.dat") << net;

        cout << "Training is completed." << endl;


    }
    catch (std::exception& e)
    {
        cout << "*****EXCEPTION*****" << endl;
        cout << e.what() << endl;
        cout << "*******************" << endl;
    }
}
/*********************************************************************************************************************************************************/
void train_shape_predictor(const std::string netFile, const std::string xmlFile)
{
    using namespace std;
    using namespace dlib;

    string shapePredictorDatasetFile = xmlFile + "sp.xml";

    cout << "Getting detection rectangles." << endl;
    std::vector<std::vector<mmod_rect>> detections = get_detected_rectanges(netFile, xmlFile);

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
    try
    {
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

        net_type net;
        cout << "Deserializing net." << endl;
        deserialize(netFile) >> net;

        cout << "Serializing net and shape predictor ." << endl;
        serialize(netFile) << net << sp;

    }
    catch (std::exception& e)
    {
        cout << "ERROR" << endl;
        cout << e.what() << endl;
    }

}
/*********************************************************************************************************************************************************/
void train_state(const std::string trainFile, const std::string outFile)
{
    using namespace dlib;
    using namespace std;

    try
    {
        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, trainFile);


        cout << "Number of training images: " << trainingImages.size() << endl;

        mmod_options options(trainingBoxes, STATE_WINDOW_WIDTH, STATE_WINDOW_HEIGHT);

       	//options.overlaps_ignore = test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD);

        cout << "Number of detector windows " << options.detector_windows.size() << endl;
        state_net_type net(options);

        net.subnet().layer_details().set_num_filters(options.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<state_net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<state_net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif

        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);

        trainer.set_iterations_without_progress_threshold(STATE_ITERATION_WITHOUT_PROGRESS_THRESHOLD);

        trainer.set_synchronization_file("STATE_SYNC", std::chrono::minutes(SYNC_INTERVAL));
	std::cout << "Max epoch count: " << trainer.get_max_num_epochs() << std::endl;
        trainer.train(trainingImages, trainingBoxes);
/*
        random_cropper cropper;

        //rows then cols
        cropper.set_chip_dims(STATE_CHIP_HEIGHT, STATE_CHIP_WIDTH);
        //cropper.set_min_object_size(MIN_OBJECT_SIZE_L, MIN_OBJECT_SIZE_S);
        cropper.set_max_object_size(0.5f);

        cropper.set_background_crops_fraction(BACKGROUND_CROP_FRACTION);

        //defaulted to false
        cropper.set_randomly_flip(false);
        cropper.set_max_rotation_degrees(0);
        cropper.set_translate_amount(0);

        dlib::rand rnd;

        cout << trainer << cropper << endl;

        std::vector<matrix<rgb_pixel>>      miniBatchImages;
        std::vector<std::vector<mmod_rect>> miniBatchLabels;
#ifdef WINDOW
	image_window win;
#endif
        while (trainer.get_learning_rate() >= MINIMAL_LEARNING_RATE)
        {
            cropper(STATE_BATCH_SIZE, trainingImages, trainingBoxes, miniBatchImages, miniBatchLabels);
	

            for (matrix<rgb_pixel> &img : miniBatchImages)
		{
                	//convert_to_grayscale(img);
                	disturb_colors(img, rnd);
#ifdef WINDOW
			win.set_image(img);
			cin.get();
#endif
		}


            trainer.train_one_step(miniBatchImages, miniBatchLabels);
        }
*/
        trainer.get_net();
        net.clean();

        std::string serializationFile = "state_net.dat";
        if (outFile.length() != 0)
            serializationFile = outFile;

        serialize(serializationFile) << net;

        cout << "Training is completed." << endl;
        cout << "Training results: " << test_object_detection_function(net, trainingImages, trainingBoxes) << endl;

    }
    catch (std::exception& e)
    {
        cout << "*****EXCEPTION*****" << endl;
        cout << e.what() << endl;
        cout << "*******************" << endl;
    }
}
/*********************************************************************************************************************************************************/
void train_resnet(const std::string trainFile)
{
    using namespace std;
    using namespace dlib;


    try
    {

        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, trainFile);

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

        resnet_net_type net(options);

        net.subnet().layer_details().set_num_filters(options.detector_windows.size());
        //std::cout << net << std::endl;
        //std::cout << layer<tag1>(net) << std::endl;

#ifdef MULTIPLE_GPUS
        dnn_trainer<resnet_net_type, sgd> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<resnet_net_type, sgd> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
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
    catch (std::exception& e)
    {
        cout << "*****EXCEPTION*****" << endl;
        cout << e.what() << endl;
        cout << "*******************" << endl;
    }

}
