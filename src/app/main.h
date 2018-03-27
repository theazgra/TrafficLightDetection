/** \file main.h
 * Entry point.
 */

/*! \mainpage Main page
 * Most things can be found in Files tab.
 * 
 * For best performace build with cuDNN and AVX instructions (for Dlib).
 * 
 * Application support -h or --help parameter to display help in console.
 */

#ifndef TLD_MAIN_H
#define TLD_MAIN_H

#include "cropper_test.h"
#include "traffic_light_train.h"
#include "traffic_light_test.h"
#include "cv_utils.h"
#include "extern_files/args.hxx"
#include "location_detector_trainer.h"
#include "state_detector_trainer.h"

/// Namespace used to avoid collisions with file_exists
namespace file
{
    /// Check if file exists.
    /// \param File name.
    /// \return True if file exists.
    bool file_exists(const std::string& file)
    {
        if (file.length() == 0)
            return false;

        std::ifstream infile(file);
        return infile.good();
    }

    /// Check if all files exist.
    /// \param fileNames File names.
    /// \return True if all files exist.
    bool files_exist(const std::vector<std::string>& fileNames)
    {
        for(const std::string& fileName : fileNames)
        {
            if (!file_exists(fileName))
            {
                std::cout << "File " << fileName << " does not exist!";
                return false;
            }
        }

        return true;
    }
}

/// Entry point of application, parses command line options.
/// \param argc Argument count.
/// \param argv Arguments.
/// \return Program exit code.
int main(int argc, const char* argv[]);

/// Start training of CNN.
/// \param xmlFile XML file of data annotatations.
/// \param xmlFile2 Second XML file of data annotatations.
/// \param resnet If ResNet should be used.
/// \return Exit code.
int start_train(const std::string &xmlFile, bool resnet, const std::string &xmlFile2 = "");


/// Start testing of CNN.
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param xmlFile XML file of data annotatations.
/// \param display Err If only error in detection should be displayed.
/// \param display If results should be displayed.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_test(const std::string &netFile, const std::string &stateNetFile, const std::string &xmlFile, bool displayErr,
               bool display, bool resnet);

/// Call to measure accurace of model, uses F One scoring
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param xmlFile XML file of data annotatations.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_f_one(const std::string &netFile, const std::string &stateNetFile, const std::string &xmlFile, bool resnet);

/// Train state predicting CNN.
/// \param xmlFile XML file of data annotatations.
/// \param outFile File to which serialize results.
/// \return Exit code.
int start_train_state(const std::string &xmlFile, const std::string &outFile);

/// Train shape predictor, to improve traffic light detection.
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_train_sp(const std::string &netFile, const std::string &xmlFile, bool resnet);

/// Start cropper test.
/// \param xmlFile XML file of data annotatations.
/// \param display If crops should be displayed
/// \return Exit code.
int start_cropper_test(const std::string &xmlFile, bool display);

/// Test the state detection.
/// \param netFile Serialized state network.
/// \param imgFile Input image file.
/// \return Exit code.
int start_detect_state(const std::string &netFile, const std::string &imgFile);

/// Start visualization.
/// \param netFile Serialized network.
/// \param imgFile Input image file.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_visualize(const std::string &netFile, const std::string &imgFile, bool resnet);

/// Start saving crops.
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_crops(const std::string &netFile, const std::string &stateNetFile, const std::string &xmlFile,
                const std::string &outFolder, bool resnet);

/// Start saving sized crops. Size is defined in app_settings.xml
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_sized_crops(const std::string &netFile, const std::string &stateNetFile, const std::string &xmlFile,
                      const std::string &outFolder, bool resnet);

/// Start processing video file.
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param videoFile Input video file.
/// \param outFolder Output folder, where to save results.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_video(const std::string &netFile, const std::string &stateNetFile, const std::string &videoFile,
                const std::string &outFolder, bool resnet);


/// Start processing frames from xml file and using shape predictor.
/// \param netFile Serialized network.
/// \param stateNetFile Serialized state network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \param stateNetFile Serialized state network.
/// \param resnet Use ResNet model.
/// \return Exit code.
int start_video_frames(const std::string &netFile, const std::string &stateNetFile, const std::string &outFolder,
                       const std::string &xmlFile, bool resnet);


#endif //TLD_MAIN_H
