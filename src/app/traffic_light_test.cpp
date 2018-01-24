#include "traffic_light_test.h"

using namespace std;
using namespace dlib;

void test(std::string netFile, std::string testFile, TestType testType, bool saveImages)
{
    net_type net;
    deserialize(netFile) >> net;

    std::vector<matrix<rgb_pixel>> testImages;
    std::vector<std::vector<mmod_rect>> boxes;
    load_image_dataset(testImages, boxes, testFile);

    mmod_rect m;
    mmod_options options(boxes, DW_LONG_SIDE, DW_SHORT_SIDE);

    if (testType == FullTest || testType == NoDisplay)
    {
        dlib::matrix<double, 1, 3> testResult = test_object_detection_function(net, testImages, boxes, test_box_overlap(), 0, options.overlaps_ignore);

        cout << "Precision:                 " << testResult(0) << endl;
        cout << "Fraction of found objects: " << testResult(1) << endl;
        cout << "Average precision:         " << testResult(2) << endl << endl;

        cout << "Precision: 1 means no false alarms, 0 means all hits were false alarms." << endl;
        cout << "Fraction: 	1 means all targets were found, 0 mean that detector did not locate any object." << endl;
        cout << "Average: 	Overall quality of the detector.." << endl;
    }


    if (testType == FullTest || testType == DisplayOnly || testType == OnlyErrorDisplay)
    {
        image_window window;

        int imgIndex = -1;
        for (matrix<rgb_pixel>& image : testImages)
        {
            ++imgIndex;
            window.clear_overlay();

            std::vector<mmod_rect> detections = net(image);

            cout << "Image #" << imgIndex << ". Ground truth: " << boxes.at(imgIndex).size()
                 << " bounding boxes. Found: " << detections.size() << " bounding boxes." << endl;


            if (testType == OnlyErrorDisplay && !detections.empty())
                continue;

            window.set_image(image);

            for (mmod_rect d : detections)
            {
                cout << "Bounding box with label: " << d.label << ". Detection confidence " << d.detection_confidence << endl;
                
                if (d.label == "r")
                    window.add_overlay(d.rect, rgb_pixel(255, 0, 0), "red");
                else if (d.label == "y") //y for orange, WTF?
                    window.add_overlay(d.rect, rgb_pixel(255, 255, 0), "orange");
                else if (d.label == "g")
			        window.add_overlay(d.rect, rgb_pixel(0, 255, 0), "green");
                else if (d.label == "s")
                    window.add_overlay(d.rect, rgb_pixel(0,255,0), "semafor");
            }

            if (saveImages)
            {
                save_png(image, "detected_" + to_string(imgIndex));
            }


            cout << "Press enter for next image." << endl;
            cin.get();

        }
    }
}
