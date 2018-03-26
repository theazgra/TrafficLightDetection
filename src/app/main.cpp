/** \file main.cpp
 * File contating entry point of out application.
 * Functions to start multiple methods are present and also argument parser from external library.
 */

#include "main.h"


using namespace std;
/*********************************************************************************************************************************************************/
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
    args::Flag _test(methodGroup, "test", "Overall testing in window. <xml> <net> <state-net> [display, display-error]", {"test"});
    args::Flag _testState(methodGroup, "test-state", "Testing state detection. <net> <file>", {"test-state"});
    args::Flag _cropper(methodGroup, "cropper", "Testing cropper settings. <xml> [display]", {'c', "cropper"});
    args::Flag _visualize(methodGroup, "visualize", "Visualizing of main CNN output. <net> <file>", {"visualize"});
    args::Flag _crops(methodGroup, "crops", "Saving detections in original size. <net> <xml> <out-folder> <state-net>", {"crops"});
    args::Flag _cropsSized(methodGroup, "sized-crops", "Saving detections in defined size. <net> <xml> <out-folder> <state-net>", {"sized-crops"});
    args::Flag _video(methodGroup, "video", "Saving frames from videl file. <net> <file> <out-folder> <state-net>", {"video"});
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
    args::Flag _resnet(parser, "resnet", "Use resnet for trainin/testing", {'r', "resnet"});

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

    if (_resnet)
        cout << "ResNet" << endl;

    if (_train)
        return start_train(xmlFile, _resnet, xmlFile2);
    if (_test)
        return start_test(netFile, stateNetFile, xmlFile, _displayErr, _display, _resnet);
    if (_trainState)
        return start_train_state(xmlFile, outFile);
    if (_trainSp)
        return start_train_sp(netFile, xmlFile, _resnet);
    if (_testState)
        return start_detect_state(netFile, file);
    if (_cropper)
        return start_cropper_test(xmlFile, _display);
    if (_visualize)
        return start_visualize(netFile, file, _resnet);
    if (_crops)
        return start_crops(netFile, stateNetFile, xmlFile, outFolder, _resnet);
    if (_cropsSized)
        return start_sized_crops(netFile, stateNetFile, xmlFile, outFolder, _resnet);
    if (_video)
        return start_video(netFile, stateNetFile, file, outFolder, _resnet);
    if (_videoFramesSp)
        return start_video_frames(netFile, stateNetFile, outFolder, xmlFile,_resnet);

    return 0;

}
/*********************************************************************************************************************************************************/
int start_train(const string &xmlFile, bool resnet, const string &xmlFile2)
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
        location_detector_trainer<net_type> TLTrainer;

        switch (TRAINING_METHOD)
        {

            case 1:
                //train(xmlFile);
            {
                TLTrainer.train_location_network(xmlFile);
            }

                break;
            case 2:
                //train(xmlFile, xmlFile2);
            {
                TLTrainer.train_location_network_with_tests(xmlFile, xmlFile2);
            }
                break;
        }
    }
    else
    {

        location_detector_trainer<resnet_net_type> TLTrainer;
        TLTrainer.train_location_network(xmlFile);
    }

    stopwatch.stop();
    cout << "Training finished after: " << stopwatch.formatted() << endl;
    return 0;
}
/*********************************************************************************************************************************************************/
int start_test(const std::string &netFile, const std::string &stateNetFile, const std::string &xmlFile, bool displayErr,
               bool display, bool resnet)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(stateNetFile))
    {
        cout << "State net file does not exist!" << endl;
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

    if (!resnet)
    {
        traffic_light_detector<test_net_type, state_test_net_type> detector;
        detector.test(netFile, stateNetFile, xmlFile, testType);
    }
    else
    {
        traffic_light_detector<resnet_test_net_type, state_test_net_type> detector;
        detector.test(netFile, stateNetFile, xmlFile, testType);
    }
    //test(netFile, xmlFile, testType);

    return 0;
}
/*********************************************************************************************************************************************************/
int start_train_sp(string &netFile, string &xmlFile, bool resnet)
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
    //train_shape_predictor(netFile, xmlFile);

    location_detector_trainer<net_type> tld;
    if (!resnet)
        tld.train_shape_predictor<test_net_type, state_test_net_type>(netFile, xmlFile);
    else
        tld.train_shape_predictor<resnet_test_net_type, state_test_net_type>(netFile, xmlFile);

    stopwatch.stop();
    cout << "Shape predictor training finished after: " << stopwatch.formatted() << endl;
    return 0;
}
/*********************************************************************************************************************************************************/
int start_train_state(string &xmlFile, string &outFile)
{
    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing testing data annotations does not exist!" << endl;
        return 1;
    }

    Stopwatch stopwatch;
    stopwatch.start();
    //train_state(xmlFile, outFile);

    state_detector_trainer<state_net_type> stateDetectorTrainer;
    stateDetectorTrainer.train_state_network(xmlFile, outFile);

    stopwatch.stop();
    cout << "State net finished training after: " << stopwatch.formatted() << endl;
    return 0;
}
/*********************************************************************************************************************************************************/
int start_cropper_test(string &xmlFile, bool display)
{
    if (!file::file_exists(xmlFile))
    {
        cout << "Xml file containing testing data annotations does not exist!" << endl;
        return 1;
    }

    test_cropper(xmlFile, true, display);
    return 0;
}
/*********************************************************************************************************************************************************/
int start_detect_state(string &netFile, string &imgFile)
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

    traffic_light_detector<test_net_type, state_test_net_type> detector;

    detector.detect_state(netFile, dlibImg);
    //detect_state(netFile, dlibImg);
    return 0;
}
/*********************************************************************************************************************************************************/
int start_visualize(string &netFile, string &imgFile, bool resnet)
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
/*********************************************************************************************************************************************************/
int start_crops(const std::string &netFile, const std::string &stateNetFile, const std::string &xmlFile,
                const std::string &outFolder, bool resnet)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(stateNetFile))
    {
        cout << "State net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }

    if (!resnet)
    {
        traffic_light_detector<test_net_type, state_test_net_type> detector;
        detector.save_detected_objects(netFile, stateNetFile, outFolder, dlib::rectangle(), xmlFile);
    }
    else
    {
        traffic_light_detector<resnet_test_net_type, state_test_net_type> detector;
        detector.save_detected_objects(netFile, stateNetFile, outFolder, dlib::rectangle(), xmlFile);
    }

    //save_detected_objects(netFile, xmlFile, outFolder);
    
    return 0;
}
/*********************************************************************************************************************************************************/
int start_sized_crops(const string &netFile, const std::string &stateNetFile, const string &xmlFile,
                      const string &outFolder, bool resnet)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(stateNetFile))
    {
        cout << "State net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }

    dlib::rectangle sizeRect(CROP_WIDTH, CROP_HEIGHT);
    if (!resnet)
    {
        traffic_light_detector<test_net_type, state_test_net_type> detector;
        detector.save_detected_objects(netFile, stateNetFile, outFolder, sizeRect, xmlFile);
    }
    else
    {
        traffic_light_detector<resnet_test_net_type, state_test_net_type> detector;
        detector.save_detected_objects(netFile, stateNetFile, outFolder, sizeRect, xmlFile);
    }
    //save_detected_objects(netFile, xmlFile, outFolder, sizeRect);
    
    return 0;
}
/*********************************************************************************************************************************************************/
int start_video(const std::string &netFile, const std::string &stateNetFile, const std::string &videoFile,
                const std::string &outFolder, bool resnet)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(stateNetFile))
    {
        cout << "State net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(videoFile))
    {
        cout << "Video file does not exist!" << endl;
        return 1;
    }

    if (!resnet)
    {
        traffic_light_detector<test_net_type, state_test_net_type> detector;
        detector.save_video(netFile, stateNetFile, videoFile, outFolder);
    }
    else
    {
        traffic_light_detector<resnet_test_net_type, state_test_net_type> detector;
        detector.save_video(netFile, stateNetFile, videoFile, outFolder);
    }

    //save_video(netFile, videoFile, outFolder);

    return 0;
}
/*********************************************************************************************************************************************************/
int start_video_frames(const std::string netFile, const std::string stateNetFile, const std::string outFolder,
                       const std::string xmlFile, bool resnet)
{
    if (!file::file_exists(netFile))
    {
        cout << "Net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(stateNetFile))
    {
        cout << "State net file does not exist!" << endl;
        return 1;
    }

    if (!file::file_exists(xmlFile))
    {
        cout << "XML does not exist!" << endl;
        return 1;
    }

    if (!resnet)
    {
        traffic_light_detector<test_net_type, state_test_net_type> detector;
        detector.save_video_frames_with_sp2(netFile, stateNetFile, xmlFile, outFolder);
    }
    else
    {
        traffic_light_detector<resnet_test_net_type, state_test_net_type> detector;
        detector.save_video_frames_with_sp2(netFile, stateNetFile, xmlFile, outFolder);
    }

    return 0;
}
