#ifndef TLD_TRAFFICLIGHTDETECTOR_H
#define TLD_TRAFFICLIGHTDETECTOR_H

#include <iostream>
#include <string>
#include "settings.h"


template <
        typename LocationTrainNetType,
        typename LocationTestNetType,
        typename StateTrainNetType,
        typename StateTestNetType
>
class Traffic_light_detector_trainer
{
private:
    LocationTrainNetType locationTrainNet;
    LocationTestNetType locationTestNet;
    StateTrainNetType stateTrainNet;
    StateTrainNetType stateTestNet;

    std::string datasetFile;
public:

    Traffic_light_detector_trainer(LocationTrainNetType& trainNet, StateTrainNetType& stateTrainNet)
    {
        this->locationTrainNet = trainNet;
        this->stateTrainNet = stateTrainNet;
    }

    Traffic_light_detector_trainer(LocationTestNetType& locationTestNet, StateTestNetType& stateTestNet)
    {
        this->locationTestNet = locationTestNet;
        this->stateTestNet = stateTestNet;
    }




    void print()
    {
        std::cout << locationTrainNet << std::endl;
        std::cout << "*********************" << std::endl;
        std::cout << locationTestNet << std::endl;
    }
};

#endif //TLD_TRAFFICLIGHTDETECTOR_H
