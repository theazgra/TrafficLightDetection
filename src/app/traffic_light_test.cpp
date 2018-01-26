#include "traffic_light_test.h"

using namespace std;
using namespace dlib;

/**
 * Get the number of not ignored label boxes.
 * @param boxes Vector of mmod_rectangles.
 * @return Number of boxes which are not ignored.
 */
int number_of_label_boxes(std::vector<dlib::mmod_rect> boxes)
{
    int count = 0;

    for (mmod_rect box : boxes)
    {
        if (!box.ignore)
            ++count;
    }

    return count;
}

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
        float overallFoundPercent = 0.0f;
        float overallConfidence = 0.0f;
        int falseDetectionCount = 0;
        for (matrix<rgb_pixel>& image : testImages)
        {
            ++imgIndex;
            window.clear_overlay();

            std::vector<mmod_rect> detections = net(image);
            int detectionCount = detections.size();

            int groundTruth = number_of_label_boxes(boxes.at(imgIndex));
            float foundPercent = ((detectionCount > groundTruth) ? (float)groundTruth : (float)detectionCount / (float)groundTruth) * 100.0f;

            if (detections.size() > groundTruth)
            {
                falseDetectionCount += detectionCount - groundTruth;
                cout << "Image " << imgIndex << " " << detectionCount - groundTruth  << " false detections!";
            }

            overallFoundPercent += foundPercent;

            cout << "Image #" << imgIndex << ". Ground truth: " << groundTruth
                 << " bounding boxes. Found: " << detectionCount << " bounding boxes.  " << foundPercent  << " %" << endl;


            //if (testType == OnlyErrorDisplay && !detections.empty())
            if (testType == OnlyErrorDisplay && (detectionCount == groundTruth))
                continue;

            window.set_image(image);

            int labelIndex = -1;
            float labelsConfidence = 0.0f;
            for (mmod_rect d : detections)
            {
                ++labelIndex;
                labelsConfidence += (float)d.detection_confidence;

                cout << "\tBounding box " << labelIndex << " with label: " << d.label << " Detection confidence " << d.detection_confidence << endl;

                if (d.label == "r")
                    window.add_overlay(d.rect, rgb_pixel(255, 0, 0), "red_" + to_string(labelIndex));
                else if (d.label == "y") //y for orange, WTF?
                    window.add_overlay(d.rect, rgb_pixel(255, 255, 0), "orange" + to_string(labelIndex));
                else if (d.label == "g")
			        window.add_overlay(d.rect, rgb_pixel(0, 255, 0), "green" + to_string(labelIndex));
                else if (d.label == "s")
                    window.add_overlay(d.rect, rgb_pixel(0,255,0), "semaphore" + to_string(labelIndex));
            }

            overallConfidence += (labelsConfidence / (float)(labelIndex + 1));

            if (saveImages)
            {
                cout << "Press s to save image, otherwise press enter for next image." << endl;
                std::string input;
                cin >> input;
                if (input == "s")
                {
                    save_png(image, "detected_" + to_string(imgIndex) + ".png");
                }
            }
            else
            {
                cout << "Press enter for next image." << endl;
                cin.get();
            }
        }
        cout << "False detections: " << falseDetectionCount << endl;
        cout << "Overall found: " << overallFoundPercent / (float)(imgIndex + 1) << " %." << endl;
        cout << "Overall confidence: " << overallConfidence / (float)(imgIndex + 1) << " %." << endl;
    }
}
