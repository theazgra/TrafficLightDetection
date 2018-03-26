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
#include "traffic_light_detector_trainer.h"

namespace file
{
    /// Check if file exists.
    /// \param file to check if exists.
    /// \return True if file exists.
    bool file_exists(std::string file)
    {
        if (file.length() == 0)
            return false;

        std::ifstream infile(file);
        return infile.good();
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
int start_train(std::string xmlFile, std::string xmlFile2 = "", bool resnet = false);

/// Start testing of CNN.
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \param display If results should be displayed.
/// \param displayErr If only error in detection should be displayed.
/// \return Exit code.
int start_test(std::string netFile, std::string xmlFile, bool display = false, bool displayErr = false);

/// Train state predicting CNN.
/// \param xmlFile XML file of data annotatations.
/// \param outFile File to which serialize results.
/// \return Exit code.
int start_train_state(std::string xmlFile, std::string outFile);

/// Train shape predictor, to improve traffic light detection.
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \return Exit code.
int start_train_sp(std::string netFile, std::string xmlFile);

/// Start cropper test.
/// \param xmlFile XML file of data annotatations.
/// \param display If crops should be displayed
/// \return Exit code.
int start_cropper_test(std::string xmlFile, bool display);

/// Test the state detection.
/// \param netFile Serialized state network.
/// \param imgFile Input image file.
/// \return Exit code.
int start_detect_state(std::string netFile, std::string imgFile);

/// Start visualization.
/// \param netFile Serialized network.
/// \param imgFile Input image file.
/// \return Exit code.
int start_visualize(std::string netFile, std::string imgFile);

/// Start saving crops.
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \return Exit code.
int start_crops(std::string netFile, std::string xmlFile, std::string outFolder);

/// Start saving sized crops. Size is defined in app_settings.xml
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \return Exit code.
int start_sized_crops(std::string netFile, std::string xmlFile, std::string outFolder);

/// Start processing video file.
/// \param netFile Serialized network.
/// \param videoFile Input video file.
/// \param outFolder Output folder, where to save results.
/// \return Exit code.
int start_video(std::string netFile, std::string videoFile, std::string outFolder);

/// Start processing frames from xml file.
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \return Exit code.
int start_video_frames(std::string netFile, std::string xmlFile, std::string outFolder);

/// Start processing frames from xml file and using shape predictor.
/// \param netFile Serialized network.
/// \param xmlFile XML file of data annotatations.
/// \param outFolder Output folder, where to save results.
/// \param stateNetFile Serialized state network.
/// \param resnet If resnet should be used.
/// \return Exit code.
int start_video_frames_sp(std::string netFile, std::string xmlFile, std::string outFolder, std::string stateNetFile, bool resnet = false);


#endif //TLD_MAIN_H
