#include "car_detector.h"

using namespace std;
using namespace dlib;

//5x5 conv layer that does 2x downsampling
template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;

//3x3 conv layer that does not do any downsampling?
template <long num_filters, typename SUBNET> using con3 = con<num_filters, 3, 3, 1, 1, SUBNET>;

//downsampler 3x con5d => 8x downsampling; relu and batch normalization in standard way?
template <typename SUBNET> using downsampler = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32, SUBNET>>>>>>>>>;

//3x3 conv layer with batch normalization and relu.
template <typename SUBNET> using rcon3 = relu<bn_con<con3<32, SUBNET>>>;

// The special input_rgb_image_pyramid
// layer causes the network to operate over a spatial pyramid, making the detector
// scale invariant.

using net_type = loss_mmod<con<1, 6, 6, 1, 1, rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

/*!
    ensures
        - Whenever two rectangles in boxes overlap, according to overlaps(), we set the
          smallest box to ignore.
        - returns the number of newly ignored boxes.
!*/
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

void car_detection(char* folderPath, bool train, bool test)
{
    try
    {
        const unsigned int MIN_BOX_SIZE = 35;

        const std::string folder = folderPath;

        std::vector<matrix<rgb_pixel>> images_train, images_test;
        std::vector<std::vector<mmod_rect>> boxes_train, boxes_test;

        cout << "Loading and checking bounding boxes." << endl;

        int num_overlapped_ignored_test = 0;
        for (auto& box : boxes_train)
        {
            num_overlapped_ignored_test += ignore_overlapped_boxes(box, test_box_overlap(0.50, 0.95));
        }

        int num_overlapped_ignored = 0;
        int num_additional_ignored = 0;

        for (auto& box : boxes_train)
        {
            num_overlapped_ignored += ignore_overlapped_boxes(box, test_box_overlap(0.50, 0.95));
            for (auto& bb : box)
            {
                if (bb.rect.width() < MIN_BOX_SIZE && bb.rect.height() < MIN_BOX_SIZE)
                {
                    bb.ignore = true;
                    ++num_additional_ignored;
                }
            }
        }

        cout << "Num overlapped ignored: " << num_overlapped_ignored << endl;
        cout << "Num additional ignored: " << num_additional_ignored << endl;
        cout << "Num overlapped ignored test: " << num_overlapped_ignored_test << endl;

        cout << "Num training images: " << images_train.size() << endl;
        cout << "Num testing images: " << images_test.size() << endl;

        //longest box side must be atleast 70px long
        //shorter box side must be atleast 30px long
        mmod_options options(boxes_train, 70, 30);

        options.overlaps_ignore = test_box_overlap(0.5, 0.95);

        net_type net(options);
        net.subnet().layer_details().set_num_filters(options.detector_windows.size());

        dnn_trainer<net_type> trainer(net, sgd(0.0001, 0.9));
        trainer.be_verbose();
        trainer.set_learning_rate(0.1);

        trainer.set_iterations_without_progress_threshold(50000);
        trainer.set_test_iterations_without_progress_threshold(1000);

        const string sync_file = "car_sync";
        trainer.set_synchronization_file(sync_file, std::chrono::minutes(5));

        const unsigned int MINI_BATCH_SIZE = 50;

        std::vector<matrix<rgb_pixel>> mini_batch_samples;
        std::vector<std::vector<mmod_rect>> mini_batch_labels;

        random_cropper cropper;
        cropper.set_seed(time(0));
        cropper.set_chip_dims(350, 350);
        cropper.set_min_object_size(0.2);
        cropper.set_max_rotation_degrees(2);

        dlib::rand rnd;

        cout << trainer << cropper << endl;

        const string NET_FILE_NAME = "mmod_car_detect.dat";

        if (train)
        {
            int cnt = 1;
            while (trainer.get_learning_rate() >= 1e-4)
            {
                //every 30 mini batch do test mini batch
                if (cnt % 30 != 0 || images_test.size() == 0)
                {
                    cropper(MINI_BATCH_SIZE, images_train, boxes_train, mini_batch_samples, mini_batch_labels);

                    for (auto& img : mini_batch_samples)
                        disturb_colors(img, rnd);

                    /*
                    *  It's a good idea to, at least once, put code here that displays the images
                       // and boxes the random cropper is generating.  You should look at them and
                       // think about if the output makes sense for your problem.  Most of the time
                       // it will be fine, but sometimes you will realize that the pattern of cropping
                       // isn't really appropriate for your problem and you will need to make some
                       // change to how the mini-batches are being generated.  Maybe you will tweak
                       // some of the cropper's settings, or write your own entirely separate code to
                       // create mini-batches.  But either way, if you don't look you will never know.
                       // An easy way to do this is to create a dlib::image_window to display the
                       // images and boxes.
                    */

                    trainer.train_one_step(mini_batch_samples, mini_batch_labels);
                }
                else
                {
                    cropper(MINI_BATCH_SIZE, images_test, boxes_test, mini_batch_samples, mini_batch_labels);

                    for (auto& img : mini_batch_samples)
                        disturb_colors(img, rnd);

                    trainer.test_one_step(mini_batch_samples, mini_batch_labels);

                }

                ++cnt;
            }

            trainer.get_net();
            cout << "DONE TRAINING" << endl;
            net.clean();

            serialize(NET_FILE_NAME) << net;

        }

        if (test)
        {
            cout << trainer << cropper << endl;

            cout << "\nsync_filename: " << sync_file << endl;
            cout << "num training images: "<< images_train.size() << endl;
            cout << "training results: " << test_object_detection_function(net, images_train, boxes_train, test_box_overlap(), 0, options.overlaps_ignore);
            // Upsampling the data will allow the detector to find smaller cars.  Recall that
            // we configured it to use a sliding window nominally 70 pixels in size.  So upsampling
            // here will let it find things nominally 35 pixels in size.  Although we include a
            // limit of 1800*1800 here which means "don't upsample an image if it's already larger
            // than 1800*1800".  We do this so we don't run out of RAM, which is a concern because
            // some of the images in the dlib vehicle dataset are really high resolution.

            upsample_image_dataset<pyramid_down<2>>(images_train, boxes_train, 1800*1800);
            cout << "training upsampled results: " << test_object_detection_function(net, images_train, boxes_train, test_box_overlap(), 0, options.overlaps_ignore);


            cout << "num testing images: "<< images_test.size() << endl;
            cout << "testing results: " << test_object_detection_function(net, images_test, boxes_test, test_box_overlap(), 0, options.overlaps_ignore);
            upsample_image_dataset<pyramid_down<2>>(images_test, boxes_test, 1800*1800);
            cout << "testing upsampled results: " << test_object_detection_function(net, images_test, boxes_test, test_box_overlap(), 0, options.overlaps_ignore);
        }

    }
    catch (std::exception& e)
    {
        cout << "Exception occured!" << endl;
        cout << e.what() << endl;
    }
}































