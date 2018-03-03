#include "traffic_light_test.h"
#define int64 opencv_broken_int
#define uint64 opencv_broken_uint

#include <dlib/opencv.h>
#include "OpenCvUtils.h"

#undef int64
#undef uint64

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
    using namespace std;
    using namespace dlib;

    test_net_type net;
    deserialize(netFile) >> net;

    std::vector<matrix<rgb_pixel>> testImages;
    std::vector<std::vector<mmod_rect>> boxes;
    load_image_dataset(testImages, boxes, testFile);

    mmod_rect m;
    mmod_options options(boxes, DW_LONG_SIDE, DW_SHORT_SIDE);

    if (testType == FullTest || testType == NoDisplay)
    {
        dlib::matrix<double, 1, 3> testResult = test_object_detection_function(net, testImages, boxes, test_box_overlap(), 0, options.overlaps_ignore);

        cout << "==============================================" << endl;
        cout << endl << "Dlib test method: " << endl;
        cout << "Precision:                 " << testResult(0) << endl;
        cout << "Fraction of found objects: " << testResult(1) << endl;
        cout << "Average precision:         " << testResult(2) << endl << endl;

        cout << "Precision: 1 means no false alarms, 0 means all hits were false alarms." << endl;
        cout << "Fraction: 	1 means all targets were found, 0 mean that detector did not locate any object." << endl;
        cout << "Average: 	Overall quality of the detector..." << endl;
        cout << "==============================================" << endl;
    }

    image_window window;

    if (testType == NoDisplay || testType == SaveCrops)
        window.close_window();


    int imgIndex = -1;
    float overallFoundPercent = 0.0f;
    int falseDetectionCount = 0;
    for (matrix<rgb_pixel>& image : testImages)
    {
        ++imgIndex;


        std::vector<mmod_rect> detections = net(image);
        int detectionCount = detections.size();

        int groundTruth = number_of_label_boxes(boxes.at(imgIndex));
        float foundPercent = (detectionCount > groundTruth) ? 100.0f : (((float)detectionCount / (float)groundTruth) * 100.0f);

        if (detections.size() > groundTruth)
        {
            falseDetectionCount += detectionCount - groundTruth;
            cout << "Image #" << imgIndex << " " << detectionCount - groundTruth  << " false detections!" << endl;
        }

        overallFoundPercent += foundPercent;

        cout << "Image #" << imgIndex << ". Ground truth: " << groundTruth
             << " bounding boxes. Found: " << detectionCount << " bounding boxes.  " << foundPercent  << " %" << endl;



        if ((testType == OnlyErrorDisplay && (detectionCount == groundTruth)) || testType == NoDisplay)
            continue;

        cv::Mat openCvImg;
        openCvImg = toMat(image);
        if (testType != SaveCrops)
        {
            window.clear_overlay();
            window.set_image(image);
        }

        int labelIndex = -1;
        for (mmod_rect d : detections)
        {

            ++labelIndex;

            if (testType == SaveCrops)
            {
                save_found_crop(openCvImg, d, labelIndex + imgIndex);
		        continue;
            }
            TLState s = Red;

            cout << "\tBounding box " << labelIndex << " with label: " << d.label << " Detection confidence " << d.detection_confidence << endl;

            if (d.label == "r")
                window.add_overlay(d.rect, rgb_pixel(255, 0, 0), "red_" + to_string(labelIndex));
            else if (d.label == "y") //y for orange, WTF?
                window.add_overlay(d.rect, rgb_pixel(255, 255, 0), "orange" + to_string(labelIndex));
            else if (d.label == "g")
                window.add_overlay(d.rect, rgb_pixel(0, 255, 0), "green" + to_string(labelIndex));
            else if (d.label == "s")
            {
                cv::Mat croppedImage = crop_image(openCvImg, d);
                window.add_overlay(d.rect, rgb_pixel(0,255,0), translate_TL_state(get_traffic_light_state(croppedImage)));
            }

        }

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
        else if (testType != SaveCrops)
        {
            cout << "Press enter for next image." << endl;
            cin.get();
        }
    }

    cout << "==============================================" << endl;
    cout << "False detections: " << falseDetectionCount << endl;
    cout << "Overall found: " << overallFoundPercent / (float)(imgIndex + 1) << " %." << endl;
    cout << "==============================================" << endl;

}

dlib::rgb_pixel get_color_for_label(std::string label)
{
    if (label == "r")
        return rgb_pixel(255, 0, 0);
    if (label == "g")
        return rgb_pixel(0, 255, 0);
    if (label == "y" || label == "o")
        return rgb_pixel(255, 127, 80);

    return rgb_pixel(200, 200, 200);
}

void save_video(std::string netFile, std::string videoFile, std::string resultFolder)
{
    using namespace dlib;
    Logger log(videoFile + "_log.txt");

    try
    {
        test_net_type net;
        deserialize(netFile) >> net;

        cv::VideoCapture videoCapture(videoFile);
        cv::Mat videoFrame;

        int frameNum = -1;
        for (;;)
        {
            ++frameNum;
            videoCapture >> videoFrame;
            if (videoFrame.empty())
            {
                break;
            }

            //Convert color to rgb space
            cv::cvtColor(videoFrame, videoFrame, CV_BGR2RGB);

            //Wrap mat with dlib wrapper.
            cv_image<rgb_pixel> dlibImg(videoFrame);

            //Assign dlib image to matrix data, because net requires matrix, or derive abstraction layer. M
            matrix<rgb_pixel> imgData;
            assign_image(imgData, dlibImg);

            std::vector<mmod_rect> detections = net(imgData);

            for (mmod_rect detection : detections)
            {
                draw_rectangle(imgData, detection.rect, get_color_for_label(detection.label));
            }

            log.write_line("Saving frame " + std::to_string(frameNum));
            save_png(imgData, resultFolder + std::to_string(frameNum) + ".png");
        }
        log.write_line("Succesfully saved all frames.");
    }
    catch (std::exception& e)
    {
        log.write_line("Error occured while saving image");
        log.write_line(e.what());
    }
}








































