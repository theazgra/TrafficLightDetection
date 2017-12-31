#include "traffic_light_test.h"

using namespace std;
using namespace dlib;

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

void test2(std::string netFile, std::string testFile)
{
//    cout << "training results: " << test_object_detection_function(net, images_train, boxes_train, test_box_overlap(), 0, options.overlaps_ignore);
	net_type net;
	deserialize(netFile) >> net;
	std::vector<matrix<rgb_pixel>> imgs;
	std::vector<std::vector<mmod_rect>> boxes;
	load_image_dataset(imgs, boxes, testFile);
	mmod_options options(boxes, DW_LONG_SIDE, DW_SHORT_SIDE);
	cout << "Training results: " << test_object_detection_function(net, imgs, boxes, test_box_overlap(), 0, options.overlaps_ignore) << endl;
}

void test(std::string netFile, std::string testFile)
{
    test2(netFile, testFile);
    return;
    net_type net;
    shape_predictor shapePredictor;
    deserialize(netFile) >> net >> shapePredictor;

    //boxes does not really have to be loaded, but can we load dataset without boxes?
    std::vector<matrix<rgb_pixel>>      testImages;
    std::vector<std::vector<mmod_rect>> testBoxes;

    load_image_dataset(testImages, testBoxes, testFile);

    image_window window;
    for (matrix<rgb_pixel>& img : testImages)
    {
        window.clear_overlay();
        window.set_image(img);
        for(auto&& mmodRect : net(img))
        {
            full_object_detection fd = shapePredictor(img, mmodRect);
            rectangle rect;
            for (unsigned int i; i << fd.num_parts(); ++i)
                rect += fd.part(i);

            if (mmodRect.label == "r") //red
                window.add_overlay(rect, rgb_pixel(255, 0, 0), mmodRect.label);
            else if (mmodRect.label == "y") //yellow
                window.add_overlay(rect, rgb_pixel(255, 255, 0), mmodRect.label);
            else if (mmodRect.label == "g") //green
                window.add_overlay(rect, rgb_pixel(0, 255, 0), mmodRect.label);


        }

        cout << "Hit enter to view the next test image.";
        cin.get();
    }

}
