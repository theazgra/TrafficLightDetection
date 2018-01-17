//
// Created by azgra on 30.12.17.
//

#include "utils.h"

using namespace std;

void display_time(double time)
{
    double min = 0;
    double sec = 0;
    double hour = 0;
    ///TODO :Logic is wrong c'mon, fix it
    /*
    hour = (time / 60) / 60;
    time -= hour * 60 * 60;
    min = time / 60;
    time-= min * 60;
    sec = time;
    */	
    //cout << hour << " hours " << min << " minutes " << sec << " seconds" << endl;
    cout << time << " seconds" << endl;
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
    cout << "  eg. --test TL_net.dat ../test/test.xml" << endl << endl;
}

