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

int thread_count = 3;
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
    }
}


int get_non_zero_count(cv::Mat & img)
{
    using namespace std;
    //std::cout << cv::format(img, cv::Formatter::FMT_NUMPY) << std::endl;
    dlib::mutex mutexLock;
    int count = 0;

    unsigned char *input = (unsigned char*)(img.data);
/*
    int r,g,b;
    for(int i = 0;i < img.cols;i++){
        for(int j = 0;j < img.rows;j++){
            b = input[img.cols * j + i ] ;
            g = input[img.cols * j + i + 1];
            r = input[img.cols * j + i + 2];

            std::cout << "[" << b << ", " << g << ", " << r << "]" << std::endl;
        }
    }
*/
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


float get_mask_coverage(TrafficLightPartInfo * partInfo)
{
/*
    cv::imshow("threshold", partInfo->resultMask);
    cv::waitKey(0);
*/

    //std::cout << partInfo->resultMask << std::endl;

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
    //std::cout << "White pixels: " << whitePixelCount << " nonZero pixels: " << nonZero << std::endl;

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

    //std::cout << cv::format(partInfo->circleMask, cv::Formatter::FMT_NUMPY) << std::endl;
    partInfo->nonZeroPixelCount = get_non_zero_count(circleMask);
    //std::cout << "Non zero pixels: " << get_non_zero_count(circleMask) << std::endl;



    resultMask = Mat1b(circleMask.rows, circleMask.cols);

    for (std::pair<Scalar, Scalar> bounds  : partInfo->hsvRanges)
    {
        //inRange(*partInfo->trafficLightPart, bounds.first, bounds.second, mask);
        inRange(partInfo->circleMask, bounds.first, bounds.second, mask);
        resultMask = resultMask | mask;

    }
    partInfo->resultMask = resultMask;


    partInfo->maskCoverage = get_mask_coverage(partInfo);

/*
    namedWindow("green");
    imshow("green", *partInfo->trafficLightPart);
    waitKey(0);
    imshow("green", partInfo->circleMask);
    waitKey(0);
*/


    dlib::auto_mutex countLock(count_mutex);
    --thread_count;
    count_signaler.signal();
}



TLState get_traffic_light_state(cv::Mat & img)
{
    using namespace cv;

    cvtColor(img, img, CV_BGR2HSV);


    Mat topPart, middlePart, bottomPart;

    int partHeight = img.rows / 3;

    topPart =       Mat(img, Rect(0,0,img.cols, partHeight));
    middlePart =    Mat(img, Rect(0,partHeight,img.cols, partHeight));
    bottomPart =    Mat(img, Rect(0,2 * partHeight,img.cols, partHeight));

    TrafficLightPartInfo topPartInfo(topPart, redBounds);
    TrafficLightPartInfo middlePartInfo(middlePart, orangeBounds);
    TrafficLightPartInfo bottomPartInfo(bottomPart, greenBounds);



    /*
    thread(&topPartInfo);
    thread(&middlePartInfo);
    thread(&bottomPartInfo);
     */


    dlib::create_new_thread(thread, &topPartInfo);
    dlib::create_new_thread(thread, &middlePartInfo);
    dlib::create_new_thread(thread, &bottomPartInfo);



    while (thread_count > 0)
    {
        count_signaler.wait();
    }




    std::cout << "red coverage: " << topPartInfo.maskCoverage << std::endl;
    std::cout << "orange coverage: " << middlePartInfo.maskCoverage << std::endl;
    std::cout << "green coverage: " << bottomPartInfo.maskCoverage << std::endl;

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


}

