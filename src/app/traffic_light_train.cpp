
#include "traffic_light_train.h"
using namespace std;
using namespace dlib;


/**
 * Taken from http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html
 * @param boxes Vector of bounding boxes.
 * @param overlaps Function which determines if two boxes overlaps.
 * @return Number of ignored boxes.
 */
int ignore_overlapped_boxes(std::vector<mmod_rect> boxes, const test_box_overlap& overlaps)
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
///testing net from "http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html"

//5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;

//3x3 conv layer that does not do any downsampling
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;

//downsampler 3x con5d => 8x downsampling; relu and batch normalization in standard way?
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;

//3x3 conv layer with batch normalization and relu.
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;

using net_type = loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;


void train(const char * folder)
{
    //"Debug" mode.
    try
    {
        std::string dataFolder = folder;

        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<matrix<rgb_pixel>>      testingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;
        std::vector<std::vector<mmod_rect>> testingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, dataFolder + "/train.xml");
        load_image_dataset(testingImages, testingBoxes, dataFolder + "/test.xml");

        cout << "Loading and checking bounding boxes." << endl;

        int num_overlapped_ignored_test = 0;
        for (std::vector<mmod_rect>& testBoxes : testingBoxes)
        {
            num_overlapped_ignored_test += ignore_overlapped_boxes(testBoxes, test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD));
        }

        int num_overlapped_ignored = 0;
        int num_additional_ignored = 0;

        for (std::vector<mmod_rect>& boxes : trainingBoxes)
        {
            num_overlapped_ignored += ignore_overlapped_boxes(boxes, test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD));

            for (mmod_rect& boundingBox : boxes)
            {
                if (boundingBox.rect.width() < MIN_BOUNDING_BOX_SIZE && boundingBox.rect.height() < MIN_BOUNDING_BOX_SIZE)
                {
                    boundingBox.ignore = true;
                    ++num_additional_ignored;
                }
            }
        }

        cout << "Number of overlapped ignored testing bounding boxes: " << num_overlapped_ignored_test << endl;

        cout << "Number of overlapped ignored training bounding boxes: " << num_overlapped_ignored << endl;
        cout << "Number of additional ignored training bounding boxes: " << num_additional_ignored << endl;


        cout << "Number of training images: " << trainingImages.size() << endl;
        cout << "Number of testing images: " << testingImages.size() << endl;

        mmod_options options(trainingBoxes, TARGET_SIZE, MIN_TARGET_SIZE);

        options.overlaps_ignore = test_box_overlap(OVERLAP_IOU, COVERED_THRESHOLD);

        cout << "Number of detector windows " << options.detector_windows.size() << endl;
        net_type net(options);

        net.subnet().layer_details().set_num_filters(options.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), {0, 1});
#else
        dnn_trainer<net_type> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif



        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);

        trainer.set_iterations_without_progress_threshold(ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_test_iterations_without_progress_threshold(TEST_ITERATION_WITHOUT_PROGRESS_THRESHOLD);


        trainer.set_synchronization_file("TL_SYNC_FILE", std::chrono::minutes(SYNC_INTERVAL));

        random_cropper cropper;
        cropper.set_seed(time(0));
        cropper.set_chip_dims(CHIP_SIZE, CHIP_SIZE);
        cropper.set_min_object_size(MIN_OBJECT_SIZE);
        cropper.set_max_object_size(MAX_OBJECT_SIZE);
        cropper.set_max_rotation_degrees(RANDOM_ROTATION_ANGLE);
        cropper.set_randomly_flip(RANDOM_FLIP);

        dlib::rand rnd;

        cout << trainer << cropper << endl;

        std::vector<matrix<rgb_pixel>>      miniBatchImages;
        std::vector<std::vector<mmod_rect>> miniBatchLabels;

        int cnt = 1;

        while (trainer.get_learning_rate() >= LEARNING_RATE_LIMIT)
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






























