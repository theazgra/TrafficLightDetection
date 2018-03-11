#include "cropper_test.h"
#include "traffic_light_train.h"
#include "traffic_light_test.h"
#include "utils.h"
#include "OpenCvUtils.h"


// ----------------------------------------------------------------------------------------
using namespace std;

namespace file
{
	bool file_exists(std::string file)
	{	
    		std::ifstream infile(file);
	    	return infile.good();
	}
}


int main(int argc, char** argv)
{
    const uint ARGCOUNT = 8;
    string arguments[ARGCOUNT];
    for (unsigned int i = 0; i < argc && i < ARGCOUNT; ++i)
        arguments[i] = argv[i];


    if (argc == 1 || arguments[1] == "--h" || arguments[1] == "--help")
    {
        display_help();
        return 0;
    }


    if (arguments[1] != "--state")
    {
        if (!load_settings("../app_settings.xml"))
            return 1;
        else
            cout << endl << "Loaded settings from xml." << endl;
    }

    if (arguments[1] == "--c" || arguments[1] == "--cropper")
    {
        string xmlFile = arguments[2];
        if (xmlFile.length() == 0)
        {
            cout << "Xml file was not specified!" << endl;
            return 1;
        }

        bool display = arguments[3] == "--display";
        bool displayOnly = arguments[3] == "--display-only";
        test_cropper(xmlFile, display, displayOnly);
        return 0;
    }

    if (arguments[1] == "--train")
    {
        string trainFile = arguments[2];
        string testFile  = arguments[3];

        if (trainFile.length() == 0 || !file::file_exists(trainFile))
        {
            cout << "Xml file containing training data annotations does not exist!" << endl;
            return 1;
        }

        cout << "Choosen training method (in xml):" << TRAINING_METHOD << endl;

        if (TRAINING_METHOD == 2)
        {
            if (testFile.length() == 0 || !file::file_exists(testFile))
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
                train(trainFile);
                break;
            case 2:
                train(trainFile, testFile);
                break;
	        case 3:
                train_myNet_type(trainFile);
                break;
        }

	stopwatch.stop();
        cout << "Training finished after: " << stopwatch.formatted() << endl;
        return 0;
    }

    if (arguments[1] == "--train-sp")
    {
        string netFile = arguments[2];
        string xmlFile  = arguments[3];

        train_shape_predictor(netFile, xmlFile);

        return 0;
    }

    if (arguments[1] == "--test")
    {
        string netFile = arguments[2];
        string testFile  = arguments[3];
        string displayArg = arguments[4];

        if (netFile.length() == 0 || !file::file_exists(netFile))
        {
            cout << "Net file does not exist!" << endl;
            return 1;
        }
        if (testFile.length() == 0 || !file::file_exists(testFile))
        {
            cout << "Xml file containing testing data annotations does not exist!" << endl;
            return 1;
        }

        TestType testType = NoDisplay;

        if (displayArg == "--display")
            testType = FullTest;
        else if (displayArg == "--display-only")
            testType = DisplayOnly;
        else if (displayArg == "--display-error")
            testType = OnlyErrorDisplay;
        else if (displayArg == "--save-crops")
            testType = SaveCrops;


        bool save = arguments[5] == "--save";

        test(netFile, testFile, testType, save);

        return 0;
    }

    if (arguments[1] == "--state")
    {
        bool verbose = arguments[3] == "--verbose";

        if (!file::file_exists(arguments[2]))
        {
            cout << "Image file does not exist!" << endl;
            return 1;
        }

        dlib::matrix<dlib::rgb_pixel> dlibImg;
        dlib::load_image(dlibImg, arguments[2]);

        cout << "Detected state: " << translate_TL_state(get_traffic_light_state2(dlibImg, verbose)) << endl;

        return 0;

    }

    if (arguments[1] == "--video")
    {
        string netFile = arguments[2];
        string videoFile  = arguments[3];
        string resultFolder = arguments[4];

        save_video(netFile, videoFile, resultFolder);

        return 0;
    }

    if (arguments[1] == "--video-frames")
    {
        string netFile = arguments[2];
        string xmlFile  = arguments[3];
        string resultFolder = arguments[4];

        save_video_frames(netFile, xmlFile, resultFolder);

        return 0;
    }

    if (arguments[1] == "--video-frames-sp")
    {
        string netFile = arguments[2];
        string xmlFile  = arguments[3];
        string resultFolder = arguments[4];

        save_video_frames_with_sp(netFile, xmlFile, resultFolder);

        return 0;
    }

    display_help();

    return 0;
}

