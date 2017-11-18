#include "mmod_detector.h"
#include "car_detector.h"

using namespace std;
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



    //mmod_train(argc, argv, train, false);
    car_detection(argv[1], train, test);


    return 0;
}