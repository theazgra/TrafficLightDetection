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
        cout << ((flooredMinutes == 1) ? " minute and " : " minutes and ")
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
    cout << "  eg. --traing ../test/train.xml {../test/test.xml}" << endl << endl;

    cout << "--test with next argument specifiing net dat file and XML file annotating test data" << endl;
    cout << "  Use --display-only to go straigt to visual testing." << endl;
    cout << "  eg. --test TL_net.dat ../test/test.xml" << endl << endl;
}

