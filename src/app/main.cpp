#include "cropper_test.h"
#include "traffic_light_train.h"
#include "traffic_light_test.h"
#include "utils.h"
#include "cv_utils.h"
#include "args.hxx"


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

int start_train(const std::string xmlFile, const std::string xmlFile2 = "", bool resnet = false);
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
int start_video_frames_sp(const std::string netFile, const std::string xmlFile, const std::string outFolder, const std::string stateNetFile, bool resnet = false);
/******************************************************************************************************************/
int main(int argc, const char* argv[])
{
    if (!load_settings("../app_settings.xml"))
        return 1;

    args::ArgumentParser parser("Traffic light detection system.",
                                "Mandatory arguments are in <>, optional arguments are in []");
    args::Group methodGroup(parser, "Methods", args::Group::Validators::AtMostOne);//Maybe xor
    args::HelpFlag _help(methodGroup, "help", "Print help", {'h', "help"});
    args::Flag _train(methodGroup, "train", "Training main CNN. <xml> <out-file> [xml2]", {"train"});
    args::Flag _trainState(methodGroup, "train-state", "Training state CNN. <xml> <out-file>", {"train-state"});
    args::Flag _trainSp(methodGroup, "train-sp", "Training of shape predictor.<xml> <net>", {"train-sp"});
    args::Flag _test(methodGroup, "test", "Overall testing in window. <xml> <net> [display, display-error]", {"test"});
    args::Flag _testState(methodGroup, "test-state", "Testing state detection. <net> <file>", {"test-state"});
    args::Flag _cropper(methodGroup, "cropper", "Testing cropper settings. <xml> [display]", {'c', "cropper"});
    args::Flag _visualize(methodGroup, "visualize", "Visualizing of main CNN output. <net> <file>", {"visualize"});
    args::Flag _crops(methodGroup, "crops", "Saving detections in original size. <net> <xml> <out-folder>", {"crops"});
    args::Flag _cropsSized(methodGroup, "sized-crops", "Saving detections in defined size. <net> <xml> <out-folder>", {"sized-crops"});
    args::Flag _video(methodGroup, "video", "Saving frames from videl file. <net> <file> <out-folder>", {"video"});
    args::Flag _videoFrames(methodGroup, "video-frames", "Saving frames from xml file. <net> <xml> <out-folder>", {"video-frames"});
    args::Flag _videoFramesSp(methodGroup, "video-frames-sp",
                              "Saving frames from xml file with help of shape predictor. <net> <state-net> <xml> <out-folder>", {"video-frames-sp"});

    args::ValueFlag<std::string> _netFile(parser, "net", "Serialized net file.", {"net"});
    args::ValueFlag<std::string> _stateNetFile(parser, "state-net", "Serialized state net file.", {"state-net"});
    args::ValueFlag<std::string> _xml(parser, "xml", "Xml file containing data annotations..", {"xml"});
    args::ValueFlag<std::string> _xml2(parser, "xml2", "Xml file containing second data annotations..", {"xml2"});
    args::ValueFlag<std::string> _file(parser, "file", "Image or video vile.", {"file"});
    args::ValueFlag<std::string> _outFile(parser, "out-file", "Output file where to serialize result.", {"out-file"});
    args::ValueFlag<std::string> _outFolder(parser, "out-folder", "Output folder.", {"out-folder"});

    //args::Group methodGroup(parser, "display arguments", args::Group::Validators::Xor);//Maybe xor
    args::Flag _display(parser, "display", "Display images.", {"display"});
    args::Flag _displayErr(parser, "display-error", "Display only errors.", {"display-error"});
    args::Flag _resnet(parser, "resner", "Use resnet for trainin/testing", {'r', "resnet"});

    try {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help) {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    std::string netFile, stateNetFile, xmlFile, xmlFile2, outFolder, outFile, file;
    netFile = _netFile ? _netFile.Get() : "";
    stateNetFile = _stateNetFile ? _stateNetFile.Get() : "";
    xmlFile = _xml ? _xml.Get() : "";
    xmlFile2 = _xml2 ? _xml2.Get() : "";
    outFile = _outFile ? _outFile.Get() : "";
    outFolder = _outFolder ? _outFolder.Get() : "";
    file = _file ? _file.Get() : "";

    if (_train)
        return start_train(xmlFile, xmlFile2);
    if (_test)
        return start_test(netFile, xmlFile, _display, _displayErr);
    if (_trainState)
        return start_train_state(xmlFile, outFile);
    if (_trainSp)
        return start_train_sp(netFile, xmlFile);
    if (_testState)
        return start_detect_state(netFile, file);
    if (_cropper)
        return start_cropper_test(xmlFile, _display);
    if (_visualize)
        return start_visualize(netFile, file);
    if (_crops)
        return start_crops(netFile, xmlFile, outFolder);
    if (_cropsSized)
        return start_sized_crops(netFile, xmlFile, outFolder);
    if (_video)
        return start_video(netFile, file, outFolder);
    if (_videoFrames)
        return start_video_frames(netFile, xmlFile, outFolder);
    if (_videoFramesSp)
        return start_video_frames_sp(netFile, xmlFile, outFolder, stateNetFile);

    return 0;

}
/******************************************************************************************************************/
int start_train(const std::string xmlFile, const std::string xmlFile2, bool resnet)
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

    if (!resnet){
        switch (TRAINING_METHOD)
        {
            case 1:
                train(xmlFile);
                break;
            case 2:
                train(xmlFile, xmlFile2);
                break;
        }
    }
    else
    {
        train_resnet(xmlFile);
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
int start_video_frames_sp(const std::string netFile, const std::string xmlFile, const std::string outFolder, const std::string stateNetFile, bool resnet)
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

    if (file::file_exists(stateNetFile))
    {
        if (!resnet)
            save_video_frames_with_sp2(netFile, stateNetFile, xmlFile, outFolder);
        else
            resnet_save_video_frames_with_sp2(netFile, stateNetFile, xmlFile, outFolder);
    }
    else
    {
        save_video_frames_with_sp(netFile, xmlFile, outFolder);
    }
    return 0;
}