#include <dlib/data_io/load_image_dataset.h>
#include <dlib/gui_widgets.h>
#include "OpenCvUtils.h"

cv::Scalar lowerWhite(0,0,225);
cv::Scalar upperWhite(255,255,255);

cv::Scalar lower_red_0(0, 100, 100);
cv::Scalar upper_red_0(15, 255, 255);
cv::Scalar lower_red_1(180 - 15, 100, 100);
cv::Scalar upper_red_1(180, 255, 255);

cv::Scalar lowerGreen(0, 140, 0);
cv::Scalar upperGreen(94, 255, 255);

cv::Scalar lowerOrange(0, 26, 40);
cv::Scalar upperOrange(67, 255, 255);

std::vector<std::pair<cv::Scalar, cv::Scalar>> redBounds = {
        std::pair<cv::Scalar, cv::Scalar>(lower_red_0, upper_red_0),
        std::pair<cv::Scalar, cv::Scalar>(lower_red_1, upper_red_1),
        std::pair<cv::Scalar, cv::Scalar>(lowerWhite, upperWhite)
};

std::vector<std::pair<cv::Scalar, cv::Scalar>> orangeBounds = {
        std::pair<cv::Scalar, cv::Scalar>(lowerOrange, upperOrange),
        std::pair<cv::Scalar, cv::Scalar>(lowerWhite, upperWhite)
};

std::vector<std::pair<cv::Scalar, cv::Scalar>> greenBounds = {
        std::pair<cv::Scalar, cv::Scalar>(lowerGreen, upperGreen),
        std::pair<cv::Scalar, cv::Scalar>(lowerWhite, upperWhite)
};

int thread_count = 6;
dlib::mutex count_mutex;
dlib::signaler count_signaler(count_mutex);

std::string translate_TL_state(TLState state)
{
    switch (state)
    {
        case Red:
            return "Red";
        case Orange:
            return "Orange";
        case Green:
            return "Green";
        case RedOrange:
            return "RedOrange";
        case Inactive:
            return "Inactive";
        case Error:
            return "Error";
	default:
	    return "Error";
    }
}

TLState get_most_possible_state(std::vector<std::pair<TLState, float>> states)
{
    TLState mostPossibleState = Inactive;
    float highestBrightness = -1.0f;

    for (std::pair<TLState, float> state : states)
    {
        if (state.second > highestBrightness)
        {
            mostPossibleState = state.first;
            highestBrightness = state.second;
        }
    }

    return mostPossibleState;
}

int get_non_zero_count(cv::Mat & img)
{
    using namespace std;
    dlib::mutex mutexLock;
    int count = 0;

    uchar pixel;
    dlib::parallel_for(0, img.cols * img.rows, [&](long i){

        pixel = img.data[i];

        //bool nonZero = pixel != 0;
        if (pixel != 0)
        {
            dlib::auto_mutex lock(mutexLock);
            ++count;
        }
    });

    return count;
}

float get_average_brightness(cv::Mat & img)
{
    float brightness = 0.0f;
    uchar pixel;

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            pixel = img.at<uchar>(col, row);
            brightness += pixel;
        }
    }

    brightness = brightness / (float)(img.rows * img.cols);

    return brightness;

}

float get_mask_coverage(TrafficLightPartInfo * partInfo)
{
    long totalPixelCount = partInfo->resultMask.rows * partInfo->resultMask.cols;

    long whitePixelCount = 0;

    dlib::mutex mutexLock;

    if (!partInfo->resultMask.isContinuous())
    {
        partInfo->resultMask = partInfo->resultMask.clone();
    }

    uchar pixel;
    dlib::parallel_for(0, totalPixelCount, [&](long i){
        pixel = partInfo->resultMask.data[i];

        if (pixel == 255)
        {
            dlib::auto_mutex lock(mutexLock);
            ++whitePixelCount;
        }


    });

    int nonZero = partInfo->nonZeroPixelCount;

    float coveragePerc = ((float)(whitePixelCount)) / ((float)(nonZero));

    return coveragePerc;
}






void thread(void * param)
{
    using namespace cv;
    TrafficLightPartInfo * partInfo = (TrafficLightPartInfo*)param;


    Mat1b circleMask = Mat::zeros(partInfo->trafficLightPart.rows, partInfo->trafficLightPart.cols, CV_8UC1);
    circle(circleMask,
            Point(partInfo->trafficLightPart.cols / 2,partInfo->trafficLightPart.rows / 2),
            (int)((partInfo->trafficLightPart.cols / 4) ),
            Scalar::all(255),
            -1,
            8,
            0);

    partInfo->trafficLightPart.copyTo(partInfo->circleMask, circleMask);
    Mat1b mask, resultMask;

    partInfo->nonZeroPixelCount = get_non_zero_count(circleMask);



    resultMask = Mat1b(circleMask.rows, circleMask.cols);

    for (std::pair<Scalar, Scalar> bounds  : partInfo->hsvRanges)
    {
        inRange(partInfo->circleMask, bounds.first, bounds.second, mask);
        resultMask = resultMask | mask;

    }
    partInfo->resultMask = resultMask;


    partInfo->maskCoverage = get_mask_coverage(partInfo);

    dlib::auto_mutex countLock(count_mutex);
    --thread_count;
    count_signaler.signal();
}

void thread2(void * param)
{
    ThreadParam2 * info = (ThreadParam2*)param;

    info->result = get_average_brightness(info->imgPart);

    dlib::auto_mutex countLock(count_mutex);
    --thread_count;
    count_signaler.signal();

}


TLState get_traffic_light_state(cv::Mat & img)
{
    using namespace cv;

    auto start = std::chrono::high_resolution_clock::now();

    int xOffset = (int)(img.cols * 0.3);
    int yOffset = (int)(img.rows * 0.15);

    //Crop image to our ROI
    Rect r(xOffset, yOffset, img.cols - 2*xOffset, img.rows - 2*yOffset);
    img = img(r);

    Mat gray, hsv;

    cvtColor(img, gray, CV_BGR2GRAY);
    cvtColor(img, hsv, CV_BGR2HSV);


    Mat topPartGray, middlePartGray, bottomPartGray, topPartHsv, middlePartHsv, bottomPartHsv;

    int partHeight = img.rows / 3;

    topPartGray =       Mat(img, Rect(0,0,img.cols, partHeight));
    middlePartGray =    Mat(img, Rect(0,partHeight,img.cols, partHeight));
    bottomPartGray =    Mat(img, Rect(0,2 * partHeight,img.cols, partHeight));

    topPartHsv =       Mat(img, Rect(0,0,img.cols, partHeight));
    middlePartHsv =    Mat(img, Rect(0,partHeight,img.cols, partHeight));
    bottomPartHsv =    Mat(img, Rect(0,2 * partHeight,img.cols, partHeight));


    TrafficLightPartInfo topPartInfo(topPartHsv, redBounds);
    TrafficLightPartInfo middlePartInfo(middlePartHsv , orangeBounds);
    TrafficLightPartInfo bottomPartInfo(bottomPartHsv, greenBounds);

    ThreadParam2 tp; tp.imgPart = topPartGray;
    ThreadParam2 mp; mp.imgPart = middlePartGray;
    ThreadParam2 bp; bp.imgPart = bottomPartGray;

    dlib::create_new_thread(thread, &topPartInfo);
    dlib::create_new_thread(thread, &middlePartInfo);
    dlib::create_new_thread(thread, &bottomPartInfo);

    dlib::create_new_thread(thread2, &tp);
    dlib::create_new_thread(thread2, &mp);
    dlib::create_new_thread(thread2, &bp);


    while (thread_count > 0)
    {
        count_signaler.wait();
    }

    float topBrig = tp.result;
    float middleBrig = mp.result;
    float bottomBrig = bp.result;

    std::cout << "Top brigthness: " << topBrig << std::endl;
    std::cout << "Middle brigthness: " << middleBrig << std::endl;
    std::cout << "Bottom brigthness: " << bottomBrig << std::endl;


    TLState s = get_most_possible_state(
            {
                    std::make_pair(Red, topBrig),
                    std::make_pair(Orange, middleBrig),
                    std::make_pair(Green, bottomBrig)
            });

    std::cout << "red coverage: " << topPartInfo.maskCoverage << std::endl;
    std::cout << "orange coverage: " << middlePartInfo.maskCoverage << std::endl;
    std::cout << "green coverage: " << bottomPartInfo.maskCoverage << std::endl;

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timeElapsed = finish - start;
    std::cout << "State found after: " << timeElapsed.count() << " ms." << std::endl;



    return s;
/*
    float coverageThreshold = 0.2f;

    bool redState = topPartInfo.maskCoverage >= coverageThreshold;
    bool orangeState = middlePartInfo.maskCoverage >= coverageThreshold;
    bool greenState = bottomPartInfo.maskCoverage >= coverageThreshold;

    if (redState && !orangeState && !greenState)
        return Red;
    if (redState && orangeState && !greenState)
        return RedOrange;
    if (orangeState && !redState && !greenState)
        return Orange;
    if (greenState && !redState && !orangeState)
        return Green;
    if (!redState && !orangeState & !greenState)
        return Inactive;


    return TLState::Error;
*/

}


void train_state_detection(const std::string file = "")
{
    using namespace dlib;

    using net_type = loss_multiclass_log<
    fc<2,
            relu<fc<84,
            relu<fc<120,
            max_pool<2,2,2,2,relu<con<16,5,5,1,1,
            max_pool<2,2,2,2,relu<con<6,5,5,1,1,
            input<matrix<rgb_pixel>>
    >>>>>>>>>>>>;

    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;

    matrix<rgb_pixel> img;
    load_image(img, "red69.png");

    images.push_back(img);
    load_image(img, "green69.png");

    images.push_back(img);

    image_window win;
    win.set_image(images[0]);
    std::cin.get();

    win.set_image(images[1]);
    std::cin.get();


    labels.push_back(0);
    labels.push_back(1);

    net_type stateNet;
    dnn_trainer<net_type> stateTrainer(stateNet);

    stateTrainer.set_min_learning_rate(0.001);
    stateTrainer.set_learning_rate(0.1);
    stateTrainer.be_verbose();

    stateTrainer.set_synchronization_file("state_net_sync", std::chrono::seconds(30));

    stateTrainer.train(images, labels);

    stateNet.clean();
    serialize("state_net.dat") << stateNet;




}

void save_found_crop(cv::Mat & mat, dlib::mmod_rect rectangle, int index)
{
    cv::Rect roi(rectangle.rect.left(), rectangle.rect.top(), rectangle.rect.width(), rectangle.rect.height());

    std::cout << roi << std::endl;

    cv::Mat cropped = mat(roi);
    std::cout << "Saving: crop_" << std::to_string(index) << ".png" << std::endl;
    cv::imwrite("crops/crop_" + std::to_string(index) + ".png" , cropped);
}




















