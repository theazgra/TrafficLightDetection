#include "traffic_light_test.h"
#define int64 opencv_broken_int
#define uint64 opencv_broken_uint

#include <dlib/opencv.h>
#include "OpenCvUtils.h"

#undef int64
#undef uint64

//width used for drawing rectangle
unsigned int RECT_WIDTH = 2;
const float scale_factor = 2.0f;

/**
 * Get the number of not ignored label boxes.
 * @param boxes Vector of mmod_rectangles.
 * @return Number of boxes which are not ignored.
 */
int number_of_label_boxes(std::vector<dlib::mmod_rect> boxes)
{
    int count = 0;

    for (mmod_rect box : boxes)
    {
        if (!box.ignore)
            ++count;
    }

    return count;
}

void test(std::string netFile, std::string testFile, TestType testType)
{
    using namespace std;
    using namespace dlib;

    test_net_type net;
    shape_predictor sp;
    deserialize(netFile) >> net >> sp;

    std::vector<matrix<rgb_pixel>> testImages;
    std::vector<std::vector<mmod_rect>> boxes;
    load_image_dataset(testImages, boxes, testFile);
/*
    mmod_rect m;
    mmod_options options(boxes, DW_LONG_SIDE, DW_SHORT_SIDE);

    if (testType == FullTest || testType == NoDisplay)
    {
        dlib::matrix<double, 1, 3> testResult = test_object_detection_function(net, testImages, boxes, test_box_overlap(), 0, options.overlaps_ignore);

        cout << "==============================================" << endl;
        cout << endl << "Dlib test method: " << endl;
        cout << "Precision:                 " << testResult(0) << endl;
        cout << "Fraction of found objects: " << testResult(1) << endl;
        cout << "Average precision:         " << testResult(2) << endl << endl;

        cout << "Precision: 1 means no false alarms, 0 means all hits were false alarms." << endl;
        cout << "Fraction: 	1 means all targets were found, 0 mean that detector did not locate any object." << endl;
        cout << "Average: 	Overall quality of the detector..." << endl;
        cout << "==============================================" << endl;
    }
*/
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

        scaledImage = matrix<rgb_pixel>(image.nr()*scale_factor, image.nc()*scale_factor);
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

        cv::Mat openCvImg;
        openCvImg = toMat(scaledImage);

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

            if (!valid_rectangle(spImprovedRect, openCvImg))
            {
                std::cout << ("Wrong rectangle: L: " +
                                  std::to_string(spImprovedRect.left()) + ";T: " +
                                  std::to_string(spImprovedRect.top()) + "; W: " +
                                  std::to_string(spImprovedRect.width()) + ";H:" +
                                  std::to_string(spImprovedRect.height())) << std::endl;

                continue;
            }

            cout << "\tBounding box " << labelIndex << " with label: " << detection.label << " Detection confidence " << detection.detection_confidence << endl;

            if (detection.label == "r")
                window.add_overlay(detection.rect, rgb_pixel(255, 0, 0), "red_" + to_string(labelIndex));
            else if (detection.label == "y") //y for orange, WTF?
                window.add_overlay(detection.rect, rgb_pixel(255, 255, 0), "orange" + to_string(labelIndex));
            else if (detection.label == "g")
                window.add_overlay(detection.rect, rgb_pixel(0, 255, 0), "green" + to_string(labelIndex));
            else if (detection.label == "s")
            {
                cv::Mat croppedImage = crop_image(openCvImg, spImprovedRect);
                window.add_overlay(detection.rect, rgb_pixel(0,255,0), translate_TL_state(get_traffic_light_state(croppedImage)));
            }

        }
    }

    cout << "==============================================" << endl;
    cout << "False detections: " << falseDetectionCount << endl;
    cout << "Overall found: " << overallFoundPercent / (float)(imgIndex + 1) << " %." << endl;
    cout << "==============================================" << endl;

}

void save_video(std::string netFile, std::string videoFile, std::string resultFolder)
{
    using namespace dlib;
    Logger logger(videoFile + "_log.txt");

    try
    {
        test_net_type net;
        deserialize(netFile) >> net;

        cv::VideoCapture videoCapture(videoFile);
        cv::Mat videoFrame, croppedImage;
        matrix<rgb_pixel> scaledImage;

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

            scaledImage = matrix<rgb_pixel>(imgData.nr() * scale_factor, imgData.nc() * scale_factor);
            //resize up
            resize_image(imgData, scaledImage);

            videoFrame = toMat(scaledImage);

            std::vector<mmod_rect> detections = net(scaledImage);

            for (mmod_rect& detection : detections)
            {
                croppedImage = crop_image(videoFrame, detection);

                draw_rectangle(scaledImage,
                               detection.rect,
                               get_color_for_state(get_traffic_light_state(croppedImage)),
                               RECT_WIDTH);
            }

            //resize back down.
            resize_image(scaledImage, imgData);

            logger.write_line("Saving frame " + std::to_string(frameNum));
            save_png(imgData, resultFolder +"/"+ std::to_string(frameNum) + ".png");
        }
        logger.write_line("Succesfully saved all frames.");
    }
    catch (std::exception& e)
    {
        logger.write_line("Error occured while saving image");
        logger.write_line(e.what());
    }
}

void save_video_frames(std::string netFile, std::string xmlFile, std::string resultFolder)
{
    using namespace std;
    using namespace dlib;

    Logger logger(xmlFile + "_log.txt");

    test_net_type net;
    deserialize(netFile) >> net;

    std::vector<matrix<rgb_pixel>> videoFrames;
    std::vector<std::vector<mmod_rect>> boxes;

    load_image_dataset(videoFrames, boxes, xmlFile);
    boxes.clear();

    int frameNum = -1;
    cv::Mat openCvImg, croppedImage;
    matrix<rgb_pixel> scaledFrame;
    Stopwatch stopwatch;

    for (matrix<rgb_pixel>& frame : videoFrames)
    {
	    stopwatch.start();
        ++frameNum;

    	scaledFrame = matrix<rgb_pixel>(frame.nr() * scale_factor, frame.nc() * scale_factor);

	    resize_image(frame, scaledFrame);
        openCvImg = toMat(scaledFrame);

        logger.write_line("Processing frame " + std::to_string(frameNum));

        std::vector<mmod_rect> detections = net(scaledFrame);
	
        for (mmod_rect& detection : detections)
        {
	        croppedImage = crop_image(openCvImg, detection);
            draw_rectangle(scaledFrame,
                           detection.rect,
                           get_color_for_state(get_traffic_light_state(croppedImage)),
                           RECT_WIDTH);
        }

        //resize image back
        resize_image(scaledFrame, frame);
	    stopwatch.stop();
	    logger.write_line("Time needed for frame: " + std::to_string(stopwatch.elapsed_milliseconds()) + " ms.");

        std::string fileName = resultFolder + "/" + std::to_string(frameNum) + ".png";
        logger.write_line("Saving frame " + fileName);
        save_png(frame, fileName);
    }

    logger.write_line("Succesfully saved all frames.");
}

void save_video_frames_with_sp(std::string netFile, std::string xmlFile, std::string resultFolder)
{
    //TODO: (Moravec) count states to remove flickering.
    using namespace std;
    using namespace dlib;

    cout << "Saving frames with help of shape predictor." << endl;

    Logger logger(xmlFile + "_log.txt");
    try
    {
        test_net_type net;
        shape_predictor sp;
        deserialize(netFile) >> net >> sp;

        std::vector<matrix<rgb_pixel>> videoFrames;
        std::vector<std::vector<mmod_rect>> boxes;

        load_image_dataset(videoFrames, boxes, xmlFile);
        boxes.clear();

        int frameNum = -1;

        cv::Mat openCvImg, croppedImage;
        matrix<rgb_pixel> scaledFrame;
        Stopwatch stopwatch;
        int netStopwatch = stopwatch.get_next_stopwatch_id();
        int stateStopwatch = stopwatch.get_next_stopwatch_id();

        for (matrix<rgb_pixel>& frame : videoFrames)
        {
            stopwatch.start();
            ++frameNum;

            scaledFrame = matrix<rgb_pixel>(frame.nr() * scale_factor, frame.nc() * scale_factor);

            resize_image(frame, scaledFrame);
            openCvImg = toMat(scaledFrame);

            logger.write_line("Processing frame " + std::to_string(frameNum));

	        stopwatch.start(netStopwatch);
            std::vector<mmod_rect> detections = net(scaledFrame);
	        stopwatch.stop(netStopwatch);

            stopwatch.start(stateStopwatch);

            for (mmod_rect& detection : detections)
            {
                full_object_detection fullObjectDetection = sp(scaledFrame, detection);

                rectangle spImprovedRect;
                for(unsigned long i = 0; i < fullObjectDetection.num_parts(); ++i)
                    spImprovedRect += fullObjectDetection.part(i);

                if (!valid_rectangle(spImprovedRect, openCvImg))
                {
                    logger.write_line("Wrong rectangle: L: " +
                                      std::to_string(spImprovedRect.left()) + ";T: " +
                                      std::to_string(spImprovedRect.top()) + "; W: " +
                                      std::to_string(spImprovedRect.width()) + ";H:" +
                                      std::to_string(spImprovedRect.height())) ;

                    continue;
                }

                croppedImage = crop_image(openCvImg, spImprovedRect);
                TLState detectedState = get_traffic_light_state(croppedImage);
                cout << "Detected state: " << translate_TL_state(detectedState) << "; for rectangle: " << spImprovedRect << endl;
                draw_rectangle(scaledFrame,
                               spImprovedRect,
                               get_color_for_state(detectedState),
                               RECT_WIDTH);
            }

            resize_image(scaledFrame, frame);
            stopwatch.stop(stateStopwatch);

	        logger.write_line("Time for network pass: " + std::to_string(stopwatch.elapsed_milliseconds(netStopwatch)) + " ms");
            logger.write_line("Time for state detection: " + std::to_string(stopwatch.elapsed_milliseconds(stateStopwatch)) + " ms.");

            std::string fileName = resultFolder + "/" + std::to_string(frameNum) + ".png";
            logger.write_line("Saving frame " + fileName);
            save_png(frame, fileName);
        }

        logger.write_line("Succesfully saved all frames.");
    }
    catch (std::exception& e)
    {
        logger.write_line(e.what());
    }
}

std::vector<std::vector<dlib::mmod_rect>> get_detected_rectanges(const std::string netFile, const std::string xmlFile)
{
    using namespace dlib;
    using namespace std;
    std::vector<std::vector<mmod_rect>> detections;

    test_net_type net;
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

void visualize_detection(std::string netFile, std::string imgFile)
{
	using namespace dlib;
	test_net_type net;
	shape_predictor sp;

	deserialize(netFile) >> net >> sp;

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
}

TLState detect_state(const std::string netFile, dlib::matrix<dlib::rgb_pixel> dlibImg)
{
    using namespace dlib;
    state_test_net_type net;
    deserialize(netFile) >> net;

    std::vector<mmod_rect> dets = net(dlibImg);


    image_window window;

    window.clear_overlay();
    window.set_image(dlibImg);

    for (mmod_rect& rect : dets)
        window.add_overlay(rect, rgb_pixel(10,10,10));

    std::cin.get();

    return Inactive;
}


void save_detected_objects(const std::string netFile, const std::string xmlFile, const std::string folderPath, dlib::rectangle sizeRect)
{
    using namespace std;
    using namespace dlib;

    cout << "Saving detected objects." << endl;

    try
    {
        test_net_type net;
        shape_predictor sp;
        deserialize(netFile) >> net >> sp;

        std::vector<matrix<rgb_pixel>> images;
        std::vector<std::vector<mmod_rect>> boxes;

        load_image_dataset(images, boxes, xmlFile);
        boxes.clear();

        int frameNum = -1;

        matrix<rgb_pixel> scaledFrame;
        cv::Mat matImg;

        for (matrix<rgb_pixel>& frame : images)
        {
            ++frameNum;

            scaledFrame = matrix<rgb_pixel>(frame.nr() * scale_factor, frame.nc() * scale_factor);

            resize_image(frame, scaledFrame);

            matImg = toMat(scaledFrame);

            std::vector<mmod_rect> detections = net(scaledFrame);

            int labelIndex = -1;
            for (mmod_rect& detection : detections)
            {
                ++labelIndex;

                full_object_detection fullObjectDetection = sp(scaledFrame, detection);

                rectangle spImprovedRect;
                for(unsigned long i = 0; i < fullObjectDetection.num_parts(); ++i)
                    spImprovedRect += fullObjectDetection.part(i);

                if (valid_rectangle(spImprovedRect, matImg))
                {
                    std::string fileName = folderPath + "/crop_" + std::to_string(frameNum) + "_" + std::to_string(labelIndex) + ".png";
                    save_found_crop(matImg, spImprovedRect, fileName, sizeRect);
                }
            }
        }

        cout << "Succesfully saved all frames." << endl;
    }
    catch (std::exception& e)
    {
        cout << e.what() << endl;
    }
}

bool valid_rectangle(const dlib::rectangle& rect, const dlib::matrix<dlib::rgb_pixel>& img)
{
    if (rect.left() < 0 || rect.width() < 0 || rect.top() < 0 || rect.height() < 0)
        return false;

    if (rect.right() > img.nc() || rect.width() > img.nc() || rect.bottom() > img.nr() || rect.height() > img.nr())
        return false;

    return true;
}

bool valid_rectangle(const dlib::rectangle& rect, const cv::Mat& img)
{
    if (rect.left() < 0 || rect.width() < 0 || rect.top() < 0 || rect.height() < 0)
        return false;

    if (rect.right() > img.cols || rect.width() > img.cols || rect.bottom() > img.rows || rect.height() > img.rows)
        return false;

    return true;
}






































