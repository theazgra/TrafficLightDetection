#include "traffic_light_test.h"

using namespace std;
using namespace dlib;

void test(std::string netFile, std::string testFile, bool display, bool displayOnly)
{
    net_type net;
    deserialize(netFile) >> net;

    std::vector<matrix<rgb_pixel>> testImages;
    std::vector<std::vector<mmod_rect>> boxes;
    load_image_dataset(testImages, boxes, testFile);
    mmod_options options(boxes, DW_LONG_SIDE, DW_SHORT_SIDE);

    if (!displayOnly)
    {
        dlib::matrix<double, 1, 3> testResult = test_object_detection_function(net, testImages, boxes, test_box_overlap(), 0, options.overlaps_ignore);

        cout << "Precision:                 " << testResult(0) << endl;
        cout << "Fraction of found objects: " << testResult(1) << endl;
        cout << "Average precision:         " << testResult(2) << endl << endl;

        cout << "Precision: 1 means no false alarms, 0 means all hits were false alarms." << endl;
        cout << "Fraction: 	1 means all targets were found, 0 mean that detector did not locate any object." << endl;
        cout << "Average: 	Overall quality of the detector.." << endl;
    }


    if (display || displayOnly)
    {
        image_window window;

        for (matrix<rgb_pixel>& image : testImages)
        {
            window.clear_overlay();

//            pyramid_up(image); image is not sampled down, is it???
            std::vector<mmod_rect> detections = net(image);
            window.set_image(image);

            for (mmod_rect d : detections)
            {
                if (d.label == "r")
                    window.add_overlay(d.rect, rgb_pixel(255, 0, 0), "red");
                else if (d.label == "y") //y for orange, WTF?
                    window.add_overlay(d.rect, rgb_pixel(255, 255, 0), "orange");
                else if (d.label == "g")
			        window.add_overlay(d.rect, rgb_pixel(0, 255, 0), "green");
            }

            cout << "Press enter for next image." << endl;
            cin.get();
        }
    }
}
