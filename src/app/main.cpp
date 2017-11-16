#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <iostream>
#include "mmod_detector.h"

using namespace std;
using namespace dlib;
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argv[1] == "-h" || argc < 3)
    {
        cout << "Arguments needed:" << endl;
        cout << "Folder containing data." << endl;
        cout << "0 or 1 if train." << endl;
        cout << "0 or 1 if test." << endl;

        return  1;
    }


    bool train = argv[2] == "1";
    bool test = argv[3] == "1";
    cout << "argv3 is :" << argv[3] << endl;

    mmod_train(argc, argv, train, false);

    return 0;
}