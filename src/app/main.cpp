#include "cropper_test.h"
#include "traffic_light_train.h"
#include "traffic_light_test.h"
#include "utils.h"
#include "cv_utils.h"
#include "cxxopts/cxxopts.hpp"


// ----------------------------------------------------------------------------------------
using namespace std;

namespace file
{
	bool file_exists(std::string file)
	{
        if (file.length() == 0)
            return false;

        std::ifstream infile(file);
        return infile.good();
	}
}

int start_train(const std::string xmlFile, const std::string xmlFile2 = "");
int start_test(const std::string netFile, const std::string xmlFile, bool display = false, bool displayErr = false);
int start_train_state(const std::string xmlFile, const std::string outFile);
int start_train_sp(const std::string netFile, const std::string xmlFile);
int start_cropper_test(const std::string xmlFile, bool display);
int start_detect_state(const std::string netFile, const std::string imgFile);
int start_visualize(const std::string netFile, const std::string imgFile);
int start_crops(const std::string netFile, const std::string xmlFile, const std::string outFolder);
int start_sized_crops(const std::string netFile, const std::string xmlFile, const std::string outFolder);
int start_video(const std::string netFile, const std::string videoFile, const std::string outFolder);
int start_video_frames(const std::string netFile, const std::string xmlFile, const std::string outFolder);
int start_video_frames_sp(const std::string netFile, const std::string xmlFile, const std::string outFolder);
/******************************************************************************************************************/
int main(int argc, const char* argv[])
{
    if (!load_settings("../app_settings.xml"))
        return 1;

    bool _display, _displayErr;
    _display = _displayErr = false;

    cxxopts::Options options("Traffic light detection", "Settings are loaded from ../app_settings.xml");

    options.add_options()
            ("h, help", "Print help")
            ("train", "Training main CNN. <xml> <out-file> [xml2]")
            ("train-state", "Training state CNN. <xml> <out-file>")
            ("train-sp", "Training of shape predictor.<xml> <net>")
            ("test", "Overall testing in window. <xml> <net> [display, display-error]")
            ("test-state", "Testing state detection. <net> <file>")
            ("c, cropper", "Testing cropper settings. <xml> [display]")
            ("visualize", "Visualizing of main CNN output. <net> <file>")
            ("crops", "Saving detections in original size. <net> <xml> <out-folder>")
            ("sized-crops", "Saving detections in defined size. <net> <xml> <out-folder>")
            ("video", "Saving frames from videl file. <net> <file> <out-folder>")
            ("video-frames", "Saving frames from xml file. <net> <xml> <out-folder>")
            ("video-frames-sp", "Saving frames from xml file with help of shape predictor. <net> <state-net> <xml> <out-folder>");

    options.add_options("Method arguments, mandatory in <>, optional in [].")
            ("net", "Serialized net file.", cxxopts::value<string>())
            ("state-net", "Serialized state net file.", cxxopts::value<string>())
            ("xml", "Xml file containing data annotations.", cxxopts::value<string>())
            ("xml2", "Xml file containing second data annotations.", cxxopts::value<string>())
            ("file", "Image file.", cxxopts::value<string>())
            ("out-file", "Output file where to serialize results.", cxxopts::value<string>())
            ("out-folder", "Output folder.", cxxopts::value<string>())
            ("display", "Display images.", cxxopts::value<bool>(_display))
            ("display-error", "Display only error.", cxxopts::value<bool>(_displayErr));

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help({"", "Method arguments, mandatory in <>, optional in []."}) << std::endl;
            return 0;
        }

        string netFile, stateNetFile, xmlFile, xmlFile2, outFolder, outFile, file;

        netFile = result.count("net") ? result["net"].as<string>() : "" ;
        stateNetFile = result.count("state-net") ? result["state-net"].as<string>() : "" ;
        xmlFile = result.count("xml") ? result["xml"].as<string>() : "" ;
        xmlFile2 = result.count("xml2") ? result["xml2"].as<string>() : "" ;
        outFile = result.count("out-file") ? result["out-file"].as<string>() : "" ;
        outFolder = result.count("out-folder") ? result["out-folder"].as<string>() : "" ;
        file = result.count("file") ? result["file"].as<string>() : "" ;

        if (result.count("train"))
            return start_train(xmlFile, xmlFile2);
        if (result.count("test"))
            return start_test(netFile, xmlFile, _display, _displayErr);
        if (result.count("train-state"))
            return start_train_state(xmlFile, outFile);
        if (result.count("train-sp"))
            return start_train_sp(netFile, xmlFile);
        if (result.count("test-state"))
            return start_detect_state(netFile, file);
        if (result.count("cropper"))
            return start_cropper_test(xmlFile, _display);
        if (result.count("visualize"))
            return start_visualize(netFile, file);
        if (result.count("crops"))
            return start_crops(netFile, xmlFile, outFolder);
        if (result.count("sized-crops"))
            return start_sized_crops(netFile, xmlFile, outFolder);
        if (result.count("video"))
            return start_video(netFile, file, outFolder);
        if (result.count("video-frames"))
            return start_video_frames(netFile, xmlFile, outFolder);
        if (result.count("video-frames-sp"))
            return start_video_frames_sp(netFile, xmlFile, outFolder);


    }
    catch (const cxxopts::OptionException& e)
    {
        cout << "error parsing options: " << e.what() << endl << "Try: " << endl;
        cout << options.help({"", "Method arguments, mandatory in <>, optional in []."}) << endl;
        return 1;
    }

    return 0;

}
/******************************************************************************************************************/
int start_train(const std::string xmlFile, const std::string xmlFile2)
{
    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing training data annotations does not exist!" << endl;
        return 1;
    }

    cout << "Choosen training method (in xml):" << TRAINING_METHOD << endl;

    if (TRAINING_METHOD == 2)
    {
        if (!file::file_exists(xmlFile2))
        {
            cout << "Xml file containing testing data annotations does not exist!" << endl;
            cout << "This file is needed for this training method!" << endl;
            return 1;
        }
    }

    Stopwatch stopwatch;
    stopwatch.start();

    switch (TRAINING_METHOD)
    {
        case 1:
            train(xmlFile);
            break;
        case 2:
            train(xmlFile, xmlFile2);
            break;
    }

    stopwatch.stop();
    cout << "Training finished after: " << stopwatch.formatted() << endl;
    return 0;
}
/******************************************************************************************************************/
int start_test(const std::string netFile, const std::string xmlFile, bool display, bool displayErr)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing testing data annotations does not exist!" << endl;
        return 1;
    }

    TestType testType = NoDisplay;

    if (display)
        testType = Display;
    if (displayErr)
        testType = OnlyErrorDisplay;

    test(netFile, xmlFile, testType);

    return 0;
}
/******************************************************************************************************************/
int start_train_sp(const std::string netFile, const std::string xmlFile)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing testing data annotations does not exist!" << endl;
        return 1;
    }

    Stopwatch stopwatch;
    stopwatch.start();
    train_shape_predictor(netFile, xmlFile);
    stopwatch.stop();
    cout << "Shape predictor training finished after: " << stopwatch.formatted() << endl;
    return 0;
}
/******************************************************************************************************************/
int start_train_state(const std::string xmlFile, const std::string outFile)
{
    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing testing data annotations does not exist!" << endl;
        return 1;
    }

    Stopwatch stopwatch;
    stopwatch.start();
    train_state(xmlFile, outFile);
    stopwatch.stop();
    cout << "State net finished training after: " << stopwatch.formatted() << endl;
    return 0;
}
/******************************************************************************************************************/
int start_cropper_test(const std::string xmlFile, bool display)
{
    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing testing data annotations does not exist!" << endl;
        return 1;
    }

    test_cropper(xmlFile, true, display);
    return 0;
}
/******************************************************************************************************************/
int start_detect_state(const std::string netFile, const std::string imgFile)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(imgFile))
    {
        cout << "Image does not exist!" << endl;
        return 1;
    }

    dlib::matrix<dlib::rgb_pixel> dlibImg;
    dlib::load_image(dlibImg, imgFile);

    detect_state(netFile, dlibImg);
    return 0;
}
/******************************************************************************************************************/
int start_visualize(const std::string netFile, const std::string imgFile)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(imgFile))
    {
        cout << "Image does not exist!" << endl;
        return 1;
    }
    
    visualize_detection(netFile, imgFile);
    return 0;
}
/******************************************************************************************************************/
int start_crops(const std::string netFile, const std::string xmlFile, const std::string outFolder)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }
    
    save_detected_objects(netFile, xmlFile, outFolder);
    
    return 0;
}
/******************************************************************************************************************/
int start_sized_crops(const std::string netFile, const std::string xmlFile, const std::string outFolder)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }

    dlib::rectangle sizeRect(CROP_WIDTH, CROP_HEIGHT);
    save_detected_objects(netFile, xmlFile, outFolder, sizeRect);
    
    return 0;
}
/******************************************************************************************************************/
int start_video(const std::string netFile, const std::string videoFile, const std::string outFolder)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(videoFile))
    {
        cout << "Video file does not exist!" << endl;
        return 1;
    }
    save_video(netFile, videoFile, outFolder);

    return 0;
}
/******************************************************************************************************************/
int start_video_frames(const std::string netFile, const std::string xmlFile, const std::string outFolder)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }
    
    save_video_frames(netFile, xmlFile, outFolder);

    return 0;
}
/******************************************************************************************************************/
int start_video_frames_sp(const std::string netFile, const std::string xmlFile, const std::string outFolder)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }
    save_video_frames_with_sp(netFile, xmlFile, outFolder);

    return 0;
}
