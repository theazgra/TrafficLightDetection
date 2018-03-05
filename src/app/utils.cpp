//
// Created by azgra on 30.12.17.
//

#include <tgmath.h>
#include "utils.h"

using namespace std;

void display_time(double time)
{
    double minutes = 0;
    double seconds = 0;
    double hours = 0;
    
    minutes = time / 60;
    hours = minutes / 60;

    if (hours >= 1)
    {
        double flooredHours = std::floor(hours);
        minutes = (hours - flooredHours) * 60;
        double flooredMinutes = std::floor(minutes);
        seconds = std::round((minutes - flooredMinutes) * 60);

        cout << flooredHours << ((flooredHours == 1) ? " hour " : " hours ") << flooredMinutes
             << ((flooredMinutes == 1) ? " minute and " : " minutes and ")
             << seconds << ((seconds == 1) ? " second" : " seconds") << endl;
    }
    else
    {
        double flooredMinutes = std::floor(minutes);

        seconds = std::round((minutes - flooredMinutes) * 60);
        cout << flooredMinutes << ((flooredMinutes == 1) ? " minute and " : " minutes and ")
             << seconds << ((seconds == 1) ? " second" : " seconds") << endl;
    }
}

void display_help()
{
    cout << "Start this program with right arguments:" << endl;
    cout << "--h or --help to display this help" << endl << endl;

    cout << "--c or --cropper to check cropper settings" << endl;
    cout << "  Additional --display argument could be specified to display bouding boxes" << endl;
    cout << "  Use --display-only to go straigt to visual testing." << endl << endl;

    cout << "--train with next argument specifiing XML file containing data annotations" << endl;
    cout << "  if you want to train with train-test method specify xml file for testing data and change method in xml settings to 2" << endl;
    cout << "  eg. --train ../test/train.xml {../test/test.xml}" << endl << endl;

    cout << "--train-sp to train shape predictor. Pass net file and xml file." << endl;
    cout << "  This method uses dlib routine to create dataset for shape predictor" << endl << endl;



    cout << "--test with next argument specifiing net dat file and XML file annotating test data" << endl;
    cout << "  Next you can use: " << endl;
    cout << "    --display-only to go straigt to visual testing." << endl;
    cout << "    --display-error to show only images without detection or with false alarms." << endl;
    cout << "    --save to save displayed images (only used with display)." << endl;
    cout << "    --save-crops to just save detected traffic lights without displaying them." << endl << endl;

    cout << "--video to save frames from video file with detected rectangles. Pass net file, video file and folder where to save frames." << endl;
    cout << "--video-frames to save frames from xml file with detected rectangles. Pass net file, xml file and folder where to save frames." << endl;
    cout << "--video-frames-sp to save frames from xml file with detected rectangles improved by shape predictor. Pass net file, xml file and folder where to save frames." << endl << endl;

    cout << "--state and image, to test traffic light state detection." << endl;
    cout << "  --verbose to print more info." << endl << endl;

}


















