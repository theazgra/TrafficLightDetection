#include "traffic_light_test.h"

using namespace std;
using namespace dlib;


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
