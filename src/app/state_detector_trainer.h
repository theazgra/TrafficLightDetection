/** \file state_detector_trainer.h
 * Routines to train state detector.
 */

#ifndef TLD_STATE_DETECTION_TRAINER_H
#define TLD_STATE_DETECTION_TRAINER_H

#include <iostream>
#include <string>
#include "settings.h"

/// Template class for state detection trainers.
/// \tparam StateNetType Type of CNN.
template <
        typename StateNetType
>
class state_detector_trainer {

public:

    /// Train network for detection of state.
    /// \param trainFile XML file with data annotations.
    /// \param outFile Serialization file.
    void train_state_network(const std::string &stateDatasetFile, const std::string& outFile)
    {
        using namespace dlib;
        using namespace std;

        std::vector<matrix<rgb_pixel>>      trainingImages;
        std::vector<std::vector<mmod_rect>> trainingBoxes;

        load_image_dataset(trainingImages, trainingBoxes, stateDatasetFile);


        cout << "Number of training images: " << trainingImages.size() << endl;

        mmod_options mmodOptions(trainingBoxes, STATE_WINDOW_WIDTH, STATE_WINDOW_HEIGHT);
        cout << "Number of detector windows " << mmodOptions.detector_windows.size() << endl;
        StateNetType net(mmodOptions);
        net.subnet().layer_details().set_num_filters(mmodOptions.detector_windows.size());


#ifdef MULTIPLE_GPUS
        dnn_trainer<StateNetType> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM), CUDA_DEVICES);
#else
        dnn_trainer<StateNetType> trainer(net, sgd(SGD_WEIGHT_DECAY, SGD_MOMENTUM));
#endif

        trainer.be_verbose();
        trainer.set_learning_rate(LEARNING_RATE);
        trainer.set_iterations_without_progress_threshold(STATE_ITERATION_WITHOUT_PROGRESS_THRESHOLD);
        trainer.set_synchronization_file("STATE_SYNC", std::chrono::minutes(SYNC_INTERVAL));
        trainer.set_max_num_epochs(20000);

        trainer.train(trainingImages, trainingBoxes);

        trainer.get_net();
        net.clean();

        std::string serializationFile = "state_net.dat";
        if (outFile.length() != 0)
            serializationFile = outFile;

        serialize(serializationFile) << net;

        cout << "Training is completed." << endl;
        cout << "Training results: " << test_object_detection_function(net, trainingImages, trainingBoxes) << endl;
    }
};

#endif //TLD_STATE_DETECTION_TRAINER_H
