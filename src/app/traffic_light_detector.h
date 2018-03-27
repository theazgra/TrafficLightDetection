#ifndef TLD_TRAFFICLIGHTDETECTOR_H
#define TLD_TRAFFICLIGHTDETECTOR_H

#include <iostream>
#include <string>
#include "settings.h"
const int RECT_WIDTH = 3;

/// Template class for detection of traffic lights and theirs states.
/// \tparam LocationNetType Type of CNN, which is used to localize traffic lights in images.
/// \tparam StateNetType Type of CNN, which is used to detect traffic light state.
template <
        typename LocationNetType,
        typename StateNetType
>
class traffic_light_detector
{
private:

    void draw_state(dlib::matrix<dlib::rgb_pixel>& image, dlib::rectangle rect, TLState detectedState)
    {
        draw_rectangle(image, rect, get_color_for_state(detectedState), RECT_WIDTH);
    }

public:

    /// Basic test method.
    /// \param netFile Serialized network.
    /// \param stateNetFile Serialized state network.
    /// \param testFile Xml file with data annotations.
    /// \param testType Test type.
    void test(const std::string& netFile, const std::string& stateNetFile, const std::string& testFile, TestType testType)
    {
        using namespace std;
        using namespace dlib;


        LocationNetType net;
        StateNetType stateNet;
        shape_predictor sp;
        deserialize(netFile) >> net >> sp;
        deserialize(stateNetFile) >> stateNet;

        std::vector<matrix<rgb_pixel>> testImages;
        std::vector<std::vector<mmod_rect>> boxes;
        load_image_dataset(testImages, boxes, testFile);

        image_window window;

        if (testType == NoDisplay)
            window.close_window();


        int imgIndex = -1;
        float overallFoundPercent = 0.0f;
        int falseDetectionCount = 0;

        matrix<rgb_pixel> scaledImage;

        for (matrix<rgb_pixel>& image : testImages)
        {
            ++imgIndex;

            scaledImage = matrix<rgb_pixel>(image.nr()*FRAME_SCALING, image.nc()*FRAME_SCALING);
            resize_image(image, scaledImage);


            std::vector<mmod_rect> detections = net(scaledImage);
            int detectionCount = detections.size();

            int groundTruth = number_of_label_boxes(boxes.at(imgIndex));
            float foundPercent = (detectionCount > groundTruth) ? 100.0f : (((float)detectionCount / (float)groundTruth) * 100.0f);

            if (detections.size() > groundTruth)
            {
                falseDetectionCount += detectionCount - groundTruth;
                cout << "Image #" << imgIndex << " " << detectionCount - groundTruth  << " false detections!" << endl;
            }

            overallFoundPercent += foundPercent;

            cout << "Image #" << imgIndex << ". Ground truth: " << groundTruth
                 << " bounding boxes. Found: " << detectionCount << " bounding boxes.  " << foundPercent  << " %" << endl;



            if ((testType == OnlyErrorDisplay && (detectionCount == groundTruth)) || testType == NoDisplay)
                continue;

            window.clear_overlay();
            window.set_image(scaledImage);

            int labelIndex = -1;
            for (mmod_rect& detection : detections)
            {
                ++labelIndex;

                full_object_detection fullObjectDetection = sp(scaledImage, detection);

                rectangle spImprovedRect;
                for(unsigned long i = 0; i < fullObjectDetection.num_parts(); ++i)
                    spImprovedRect += fullObjectDetection.part(i);

                if (!valid_rectangle(spImprovedRect, scaledImage))
                {
                    std::cout << ("Wrong rectangle: L: " +
                                  std::to_string(spImprovedRect.left()) + ";T: " +
                                  std::to_string(spImprovedRect.top()) + "; W: " +
                                  std::to_string(spImprovedRect.width()) + ";H:" +
                                  std::to_string(spImprovedRect.height())) << std::endl;

                    continue;
                }

                cout << "\tBounding box " << labelIndex << " with label: " << detection.label << " Detection confidence " << detection.detection_confidence << endl;

                matrix<rgb_pixel> foundTrafficLight = crop_image(scaledImage, spImprovedRect);
                std::vector<mmod_rect> trafficLightDets = stateNet(foundTrafficLight);
                TLState detectedState = get_detected_state(trafficLightDets, foundTrafficLight);

                draw_state(scaledImage, spImprovedRect, detectedState);

            }

            std::cin.get();
        }

        cout << "==============================================" << endl;
        cout << "False detections: " << falseDetectionCount << endl;
        cout << "Overall found: " << overallFoundPercent / (float)(imgIndex + 1) << " %." << endl;
        cout << "==============================================" << endl;

    }
    /*********************************************************************************************************************************************************/
    /// Save processed frames from video file.
    /// \param netFile Serialized network.
    /// \param stateNetFile Serialized state network.
    /// \param videoFile Video file.
    /// \param resultFolder Where to save images.
    void save_video(const std::string& netFile, const std::string& stateNetFile, const std::string& videoFile, std::string resultFolder)
    {
        using namespace dlib;
        Logger logger(videoFile + "_log.txt");

        LocationNetType net;
        StateNetType stateNet;
        shape_predictor sp;

        deserialize(netFile) >> net >> sp;
        deserialize(stateNetFile) >> stateNet;

        cv::VideoCapture videoCapture(videoFile);
        cv::Mat videoFrame;
        matrix<rgb_pixel> scaledImage, croppedImage;

        int frameNum = -1;
        for (;;)
        {
            ++frameNum;
            videoCapture >> videoFrame;

            if (videoFrame.empty())
            {
                break;
            }

            //Convert color to rgb space
            cv::cvtColor(videoFrame, videoFrame, CV_BGR2RGB);

            //Wrap mat with dlib wrapper.
            cv_image<rgb_pixel> dlibImg(videoFrame);

            //Assign dlib image to matrix data, because net requires matrix, or derive abstraction layer. M
            matrix<rgb_pixel> imgData;
            assign_image(imgData, dlibImg);

            scaledImage = matrix<rgb_pixel>(imgData.nr() * FRAME_SCALING, imgData.nc() * FRAME_SCALING);
            //resize up
            resize_image(imgData, scaledImage);


            std::vector<mmod_rect> detections = net(scaledImage);

            for (mmod_rect& detection : detections)
            {
                full_object_detection fullObjectDetection = sp(scaledImage, detection);

                rectangle spImprovedRect;
                for(unsigned long i = 0; i < fullObjectDetection.num_parts(); ++i)
                    spImprovedRect += fullObjectDetection.part(i);

                if (!valid_rectangle(spImprovedRect, scaledImage))
                {
                    continue;
                }

                croppedImage = crop_image(scaledImage, spImprovedRect);

                matrix<rgb_pixel> foundTrafficLight = crop_image(scaledImage, spImprovedRect);
                std::vector<mmod_rect> trafficLightDets = stateNet(foundTrafficLight);
                TLState detectedState = get_detected_state(trafficLightDets, foundTrafficLight);

                draw_state(scaledImage, spImprovedRect, detectedState);
            }

            //resize back down.
            resize_image(scaledImage, imgData);

            logger.write_line("Saving frame " + std::to_string(frameNum));
            save_png(imgData, resultFolder +"/"+ std::to_string(frameNum) + ".png");
        }
        logger.write_line("Succesfully saved all frames.");

    }

/*********************************************************************************************************************************************************/
    /// Get detected object locations.
    /// \param netFile Serialized network.
    /// \param xmlFile Xml file with data annotations.
    /// \return Vector of detected objects locations.
    std::vector<std::vector<dlib::mmod_rect>> get_detected_rectanges_for_sp(const std::string& netFile, const std::string& xmlFile)
    {
        using namespace dlib;

        std::vector<std::vector<mmod_rect>> detections;

        LocationNetType net;
        deserialize(netFile) >> net;

        std::vector<matrix<rgb_pixel>> testImages;
        std::vector<std::vector<mmod_rect>> gtBoxes;

        load_image_dataset(testImages, gtBoxes, xmlFile);
        gtBoxes.clear();

        for (matrix<rgb_pixel>& testImage : testImages)
        {
            std::vector<mmod_rect> dets = net(testImage);

            detections.push_back(dets);
        }

        return detections;
    }
/*********************************************************************************************************************************************************/
    void visualize_detection(std::string imgFile)
    {
        /*
        using namespace dlib;
        LocationNetType net;
        shape_predictor sp;

        deserialize(this->locationNetFile) >> net >> sp;

        matrix<rgb_pixel> img;
        load_image(img, imgFile);

        image_window win;

        win.set_image(img);

        for (auto&& d : net(img))
        {
            auto fd = sp(img, d);
            rectangle rect;
            for (unsigned long j = 0; j < fd.num_parts(); ++j)
                rect += fd.part(j);

            win.add_overlay(rect, rgb_pixel(255,0,0));
        }

        const float lower = -2.5;
        const float upper = 0.0;
        std::cout << "jet color mapping range:  lower="<< lower << "  upper="<< upper << std::endl;

        std::vector<rectangle> rects;
        matrix<rgb_pixel> pyramid;

        using pyramid_type = std::remove_reference<decltype(input_layer(net))>::type::pyramid_type;

        create_tiled_pyramid<pyramid_type>(img, pyramid, rects,
                                           input_layer(net).get_pyramid_padding(),
                                           input_layer(net).get_pyramid_outer_padding());

        image_window winpyr(pyramid, "Tiled pyramid");

        save_png(pyramid, "tiled_pyramid.png");

        std::cout << "Number of channels in final tensor image: " << net.subnet().get_output().k() << std::endl;

        matrix<float> network_output = image_plane(net.subnet().get_output(),0,0);
        for (long k; k < net.subnet().get_output().k(); ++k)
            network_output = max_pointwise(network_output, image_plane(net.subnet().get_output(), 0, k));

        const double network_output_scale = img.nc()/(double)network_output.nc();
        std::cout << "Network output scale: " << network_output_scale << std::endl;

        resize_image(network_output_scale, network_output);

        image_window win_output(jet(network_output, upper, lower), "Output tensdor from network");

        save_png(jet(network_output, upper, lower), "output_tensor.png");

        for (long r = 0; r < pyramid.nr(); ++r)
        {
            for (long c = 0; c < pyramid.nc(); ++c)
            {
                dpoint tmp(c,r);
                tmp = input_tensor_to_output_tensor(net, tmp);
                tmp = point(network_output_scale * tmp);
                if (get_rect(network_output).contains(tmp))
                {
                    float val = network_output(tmp.y(),tmp.x());
                    // alpha blend the network output pixel with the RGB image to make our
                    // overlay.
                    rgb_alpha_pixel p;
                    assign_pixel(p , colormap_jet(val,lower,upper));
                    p.alpha = 120;
                    assign_pixel(pyramid(r,c), p);
                }
            }
        }

        image_window win_pyr_overlay(pyramid, "Detection scores on image pyramid");
        save_png(pyramid, "pyramid_score.png");

        matrix<float> collapsed(img.nr(), img.nc());
        resizable_tensor input_tensor;
        input_layer(net).to_tensor(&img, &img+1, input_tensor);
        for (long r = 0; r < collapsed.nr(); ++r)
        {
            for (long c = 0; c < collapsed.nc(); ++c)
            {
                // Loop over a bunch of scale values and look up what part of network_output
                // corresponds to the point(c,r) in the original image, then take the max
                // detection score over all the scales and save it at pixel point(c,r).
                float max_score = -1e30;
                for (double scale = 1; scale > 0.2; scale *= 5.0/6.0)
                {
                    // Map from input image coordinates to tiled pyramid coordinates.
                    dpoint tmp = center(input_layer(net).image_space_to_tensor_space(input_tensor,scale, drectangle(dpoint(c,r))));
                    // Now map from pyramid coordinates to network_output coordinates.
                    tmp = point(network_output_scale*input_tensor_to_output_tensor(net, tmp));

                    if (get_rect(network_output).contains(tmp))
                    {
                        float val = network_output(tmp.y(),tmp.x());
                        if (val > max_score)
                            max_score = val;
                    }
                }

                collapsed(r,c) = max_score;

                // Also blend the scores into the original input image so we can view it as
                // an overlay on the cars.
                rgb_alpha_pixel p;
                assign_pixel(p , colormap_jet(max_score,lower,upper));
                p.alpha = 120;
                assign_pixel(img(r,c), p);
            }
        }

        image_window win_collapsed(jet(collapsed, upper, lower), "Collapsed output tensor from the network");
        save_png(jet(collapsed, upper, lower), "collapsed_result.png");
        image_window win_img_and_sal(img, "Collapsed detection scores on raw image");
        save_png(img, "collapsed_on_image.png");

        std::cout << "Hit enter to end program" << std::endl;
        std::cin.get();
         */
    }
/*********************************************************************************************************************************************************/

    /// Detect state of traffic light.
    /// \param netFile Serialized state network.
    /// \param dlibImg Image of found traffic light.
    /// \return Detected state.
    TLState detect_state(const std::string &netFile, const dlib::matrix<dlib::rgb_pixel> dlibImg)
    {
        using namespace dlib;

        StateNetType net;
        deserialize(netFile) >> net;

        Stopwatch netPass;
        netPass.start();
        std::vector<mmod_rect> dets = net(dlibImg);
        netPass.stop();

        std::cout << "Net pass time: " << netPass.formatted() << std::endl;

        TLState  state = get_detected_state(dets, dlibImg);

        std::cout << "Detected state: " << translate_TL_state(state) << std::endl;


        image_window window;

        window.clear_overlay();
        window.set_image(dlibImg);

        for (mmod_rect& rect : dets)
            window.add_overlay(rect, rgb_pixel(255,255,255));

        std::cin.get();

        return state;
    }
/*********************************************************************************************************************************************************/

    /// Get detected state based, based on state net detections.
    /// \param detections State net detections.
    /// \param image Found traffic light image.
    /// \return Detected state.
    TLState get_detected_state(const std::vector<dlib::mmod_rect>& detections, const dlib::matrix<dlib::rgb_pixel>& image)
    {

        if (detections.empty())
            return Inactive;

        long height28Perc = (long)(0.28f * image.nr());
        long height55Perc = (long)(0.55f * image.nr());
        long height60Perc = (long)(0.6f * image.nr());

        if (detections.size() == 1)
        {
            const rectangle rect = detections.at(0).rect;

            if (rect.top() < height28Perc)
                return Red;

            if (rect.top() > height28Perc && rect.top() < height55Perc)
                return Orange;

            if (rect.top() > height60Perc)
                return Green;


        }
        else if (detections.size() == 2)
        {
            TLState state1 = get_detected_state({detections.at(0)}, image);
            TLState state2 = get_detected_state({detections.at(1)}, image);

            if ((state1 == Red && state2 == Orange) || (state1 == Orange && state2 == Red))
                return Orange;
        }

        return Ambiguous;
    }
/*********************************************************************************************************************************************************/

    /// Save detected objects.
    /// \param netFile Serialized network.
    /// \param stateNetFile Serialized state network.
    /// \param xmlFile Xml file with data annotations.
    /// \param folderPath Where to save images.
    /// \param sizeRect Optional image size.
    void save_detected_objects(const std::string& netFile, const std::string& stateNetFile, const std::string &folderPath, dlib::rectangle sizeRect,
                               const std::string &xmlFile)
    {
        using namespace dlib;

        std::cout << "Saving detected objects in parrallel" << std::endl;

        std::vector<matrix<rgb_pixel>> images;
        std::vector<std::vector<mmod_rect>> boxes;
        load_image_dataset(images, boxes, xmlFile);
        boxes.clear();

        std::vector<CudaJobInfo> cudaJobs;
        ulong cudaDeviceCount = CUDA_DEVICES.size();
        ulong deviceJobSize = images.size() / cudaDeviceCount;

        for(int index = 0; index < cudaDeviceCount; ++index)
        {
            CudaJobInfo jobInfo(CUDA_DEVICES.at(index),
                                netFile,
                                stateNetFile,
                                folderPath,
                                images,
                                (index * deviceJobSize),
                                (index == CUDA_DEVICES.size() - 1) ? images.size() : ((index * deviceJobSize) + deviceJobSize),
                                SaveCrops);


            jobInfo.sizeRectangle = sizeRect;

            cudaJobs.push_back(jobInfo);

            std::cout << "Cuda device: " << jobInfo.deviceId << std::endl;
            std::cout << "  Job image count: " << jobInfo.end - jobInfo.begin << std::endl;
        }

        dlib::parallel_for(0, cudaJobs.size(), [&](long i){
            cuda_device_process_job(cudaJobs.at(i));
        });

        double totalTime = 0;
        for (CudaJobInfo jobInfo : cudaJobs)
            totalTime += jobInfo.stopwatch.average_lap_time_in_milliseconds();

        double frameTime = totalTime / cudaJobs.size();
        double FPS = 1000 / frameTime;

        std::cout << "Average frame time across all cuda devices: " << frameTime << std::endl;
        std::cout << "Average FPS across all cuda devices: " << FPS << std::endl;

    }
/*********************************************************************************************************************************************************/
    /// Save processed from from xml file, uses shape predictor to enhance prediction and state net to detect traffic light state. (ResNet)
    /// \param netFile Serialized network.
    /// \param stateNetFile Serialized state network.
    /// \param xmlFile Xml file with data annotations.
    /// \param resultFolder Where to save images.
    void save_video_frames_with_sp2(const std::string& netFile, const std::string& stateNetFile, const std::string& xmlFile, const std::string& resultFolder)
    {
        using namespace dlib;

        std::cout << "Saving video frames with sp objects in parrallel" << std::endl;

        std::vector<matrix<rgb_pixel>> images;
        std::vector<std::vector<mmod_rect>> boxes;
        load_image_dataset(images, boxes, xmlFile);
        boxes.clear();

        std::vector<CudaJobInfo> cudaJobs;
        ulong cudaDeviceCount = CUDA_DEVICES.size();
        ulong deviceJobSize = images.size() / cudaDeviceCount;

        for(int index = 0; index < cudaDeviceCount; ++index)
        {
            CudaJobInfo jobInfo(CUDA_DEVICES.at(index),
                                netFile,
                                stateNetFile,
                                resultFolder,
                                images,
                                (index * deviceJobSize),
                                (index == CUDA_DEVICES.size() - 1) ? images.size() : ((index * deviceJobSize) + deviceJobSize),
                                SaveImages);


            cudaJobs.push_back(jobInfo);

            std::cout << "Cuda device: " << jobInfo.deviceId << std::endl;
            std::cout << "  Job image count: " << jobInfo.end - jobInfo.begin << std::endl;
        }

        dlib::parallel_for(0, cudaJobs.size(), [&](long i){
            cuda_device_process_job(cudaJobs.at(i));
        });

        double totalTime = 0;
        for (CudaJobInfo jobInfo : cudaJobs)
            totalTime += jobInfo.stopwatch.average_lap_time_in_milliseconds();

        double frameTime = totalTime / cudaJobs.size();
        double FPS = 1000 / frameTime;

        std::cout << "Average frame time across all cuda devices: " << frameTime << std::endl;
        std::cout << "Average FPS across all cuda devices: " << FPS << std::endl;

    }
/*********************************************************************************************************************************************************/
    /// Cuda device job. Processing images given in jobinfo.
    /// \param jobInfo Structure containing all needed informations.
    void cuda_device_process_job(CudaJobInfo& jobInfo)
    {
        using namespace std;
        using namespace dlib;

        std::cout << "Processing in cuda device: " << jobInfo.deviceId << std::endl;
#ifdef MULTIPLE_GPUS
        cudaSetDevice(jobInfo.deviceId);
#endif
        LocationNetType net;
        shape_predictor sp;
        StateNetType stateNet;

        deserialize(jobInfo.netFile) >> net >> sp;
        deserialize(jobInfo.stateNetFile) >> stateNet;

        ulong frameNum = jobInfo.frameIndexOffset;

        matrix<rgb_pixel> scaledFrame;


        for (ulong i = jobInfo.begin; i < jobInfo.end; ++i)
        {
            const matrix<rgb_pixel>& frame = jobInfo.jobImages.at(i);

            jobInfo.stopwatch.start_new_lap();
            ++frameNum;

            scaledFrame = matrix<rgb_pixel>(frame.nr() * FRAME_SCALING, frame.nc() * FRAME_SCALING);
            resize_image(frame, scaledFrame);

            std::vector<mmod_rect> detections = net(scaledFrame);

            uint labelIndex = 0;
            for (mmod_rect& detection : detections)
            {

                full_object_detection fullObjectDetection = sp(scaledFrame, detection);

                rectangle spImprovedRect;
                for(unsigned long i = 0; i < fullObjectDetection.num_parts(); ++i)
                    spImprovedRect += fullObjectDetection.part(i);

                if (!valid_rectangle(spImprovedRect, scaledFrame))
                    continue;

                matrix<rgb_pixel> foundTrafficLight = crop_image(scaledFrame, spImprovedRect);
                std::vector<mmod_rect> trafficLightDets = stateNet(foundTrafficLight);

                TLState detectedState = get_detected_state(trafficLightDets, foundTrafficLight);

                draw_rectangle(scaledFrame,
                               spImprovedRect,
                               get_color_for_state(detectedState),
                               RECT_WIDTH);

                if (jobInfo.jobType == SaveCrops)
                {
                    std::string fileName = jobInfo.resultFolder + "/crop_" + std::to_string(frameNum) + "_" + std::to_string(++labelIndex) + ".png";
                    save_found_crop(frame, transform_rectangle_back(spImprovedRect, FRAME_SCALING), fileName, jobInfo.sizeRectangle);
                    jobInfo.stopwatch.end_lap();
                    continue;
                }

            }
            jobInfo.stopwatch.end_lap();
            resize_image(scaledFrame, frame);

            if (jobInfo.jobType == SaveImages)
            {
                std::string fileName = jobInfo.resultFolder + "/" + std::to_string(frameNum) + ".png";
                save_png(frame, fileName);
            }
        }
    }

};

#endif //TLD_TRAFFICLIGHTDETECTOR_H
