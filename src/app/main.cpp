//#include "mmod_detector.h"
//#include "car_detector.h"
//#include "cropper_test.h"
#include "traffic_light_train.h"

using namespace std;
// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{

    //mmod_train(argc, argv, train, false);
    //car_detection(argv[1], train, test);
    //test_cropper(argv[1]);
    if (argc < 2)
    {
        cout << "Please pass folder containing training and testing data." << endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    cout << "writing to console" << endl;

    auto finish = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> timeElapsed = finish - start;
    std::cout << "Elapsed time: " << timeElapsed.count() << " s\n";


    return 0;
}