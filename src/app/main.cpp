#include "cropper_test.h"
#include "traffic_light_train.h"
#include "traffic_light_test.h"
#include "utils.h"

// ----------------------------------------------------------------------------------------
using namespace std;

bool file_exists(std::string file);
int main(int argc, char** argv)
{
    const uint ARGCOUNT = 5;
    string arguments[ARGCOUNT];
    for (unsigned int i = 0; i < argc && i < ARGCOUNT; ++i)
        arguments[i] = argv[i];


    if (argc == 1 || arguments[1] == "--h" || arguments[1] == "--help")
    {
        display_help();
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
        test_cropper(xmlFile, display);
    }

    if (arguments[1] == "--train")
    {
        string trainFile = arguments[2];
        string testFile  = arguments[3];

        if (trainFile.length() == 0 || !file_exists(trainFile))
        {
            cout << "Xml file containing training data annotations does not exist!" << endl;
            return 1;
        }

        int method = 1;
        if (arguments[4].length() != 0)
        {
            try {
                method = stoi(arguments[4].substr(2,1), nullptr, 0);
            }
            catch (std::exception)
            {
                cout << "Wrong method number. Defaulting to 1." << endl;
                method = 1;
            }
        }



        if (method == 2)
        {
            if (testFile.length() == 0 || !file_exists(testFile))
            {
                cout << "Xml file containing testing data annotations does not exist!" << endl;
                cout << "This file is needed for this learning method!" << endl;
                return 1;
            }
        }

        auto start = chrono::high_resolution_clock::now();

        switch (method)
        {
            case 1:
                train(trainFile);
                break;
            case 2:
                train(trainFile, testFile);
                break;
        }

        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> timeElapsed = finish - start;
        cout << "Training finished after: "; display_time(timeElapsed.count());
    }

    if (arguments[1] == "--test")
    {
        string netFile = arguments[2];
        string testFile  = arguments[3];
        if (netFile.length() == 0 || !file_exists(netFile))
        {
            cout << "Net file does not exist!" << endl;
            return 1;
        }
        if (testFile.length() == 0 || !file_exists(testFile))
        {
            cout << "Xml file containing testing data annotations does not exist!" << endl;
            return 1;
        }
        test(netFile, testFile);
    }

    return 0;
}

bool file_exists(std::string file)
{
    std::ifstream infile(file);
    return infile.good();
}
