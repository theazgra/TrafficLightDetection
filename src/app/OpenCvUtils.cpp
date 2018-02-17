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

bool all_zero(std::vector<float> values)
{
    bool predicate = true;

    for (float val : values)
    {
        if (val != 0)
        {
            predicate = false;
            break;
        }
    }

    return predicate;
}

bool all_below(std::vector<float> values, float threshold)
{
    bool predicate = true;

    for (float val : values)
    {
        if (val > threshold)
        {
            predicate = false;
            break;
        }
    }

    return predicate;
}

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

float get_mask_coverage(HsvTestParam * partInfo)
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

    float coveragePerc = ((float)(whitePixelCount)) / ((float)(totalPixelCount));

    return coveragePerc;
}






void thread(void * param)
{
    using namespace cv;
    HsvTestParam * partInfo = (HsvTestParam*)param;

    Mat1b mask, resultMask;

    resultMask = Mat1b(partInfo->trafficLightPart.rows, partInfo->trafficLightPart.cols);

    for (std::pair<Scalar, Scalar> bounds  : partInfo->hsvRanges)
    {
        inRange(partInfo->trafficLightPart, bounds.first, bounds.second, mask);
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
    GrayScaleTestParam * info = (GrayScaleTestParam*)param;

    info->result = get_average_brightness(info->imgPart);

    dlib::auto_mutex countLock(count_mutex);
    --thread_count;
    count_signaler.signal();

}

void show(cv::Mat & img)
{
    cv::namedWindow("Window", 0);
    cv::imshow("Window", img);
    cv::waitKey(0);
}


TLState get_traffic_light_state(cv::Mat & img)
{
    using namespace cv;

    //auto start = std::chrono::high_resolution_clock::now();

    int xOffset = (int)(img.cols * 0.3);
    int yOffset = (int)(img.rows * 0.15);

    //THRESHOLDING
    /*
    Mat kernel = (Mat_<float>(3,3) <<   1.0f,  1.0f, 1.0f,
                                        1.0f, -9.0f, 1.0f,
                                        1.0f,  1.0f, 1.0f);


    Mat imgLaplacian;
    Mat sharp = img; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    img.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    show(imgLaplacian);
    show(imgResult);
*/

    Mat thresholded;
    threshold(img, thresholded, 100, 100, CV_THRESH_TOZERO);
    show(img);
    show(thresholded);


    //Crop image to our ROI
    Rect r(xOffset, yOffset, img.cols - 2*xOffset, img.rows - 2*yOffset);

    //show(img);

    img = img(r);

    //show(img);


    Mat gray, hsv;

    cvtColor(img, gray, CV_BGR2GRAY);
    cvtColor(img, hsv, CV_BGR2HSV);

    normalize(gray, gray, 0, 100, NORM_MINMAX); // 150


    //show(normalized1);
    //show(normalized2);


    Mat topPartGray, middlePartGray, bottomPartGray, topPartHsv, middlePartHsv, bottomPartHsv;

    int partHeight = img.rows / 3;

    topPartGray =       Mat(gray, Rect(0,0,img.cols, partHeight));
    middlePartGray =    Mat(gray, Rect(0,partHeight,img.cols, partHeight));
    bottomPartGray =    Mat(gray, Rect(0,2 * partHeight,img.cols, partHeight));

    topPartHsv =       Mat(hsv, Rect(0,0,img.cols, partHeight));
    middlePartHsv =    Mat(hsv, Rect(0,partHeight,img.cols, partHeight));
    bottomPartHsv =    Mat(hsv, Rect(0,2 * partHeight,img.cols, partHeight));


    HsvTestParam hsvTop(topPartHsv, redBounds);
    HsvTestParam hsvMiddle(middlePartHsv , orangeBounds);
    HsvTestParam hsvBottom(bottomPartHsv, greenBounds);

    GrayScaleTestParam tp; tp.imgPart = topPartGray;
    GrayScaleTestParam mp; mp.imgPart = middlePartGray;
    GrayScaleTestParam bp; bp.imgPart = bottomPartGray;

#ifdef THREADING

    dlib::create_new_thread(thread, &hsvTop);
    dlib::create_new_thread(thread, &hsvMiddle);
    dlib::create_new_thread(thread, &hsvBottom);

    dlib::create_new_thread(thread2, &tp);
    dlib::create_new_thread(thread2, &mp);
    dlib::create_new_thread(thread2, &bp);


    while (thread_count > 0)
    {
        count_signaler.wait();
    }
#else
    thread(&hsvTop);
    thread(&hsvMiddle);
    thread(&hsvBottom);

    thread2(&tp);
    thread2(&mp);
    thread2(&bp);
#endif

    float topBrig = tp.result;
    float middleBrig = mp.result;
    float bottomBrig = bp.result;

    TLState grayScaleTest = get_most_possible_state(
            {
                    std::make_pair(Red, topBrig),
                    std::make_pair(Orange, middleBrig),
                    std::make_pair(Green, bottomBrig)
            });

    TLState hsvTest = get_most_possible_state(
            {
                    std::make_pair(Red, hsvTop.maskCoverage),
                    std::make_pair(Orange, hsvMiddle.maskCoverage),
                    std::make_pair(Green, hsvBottom.maskCoverage)
            });

    std::cout << "==============================================" << std::endl;
    std::cout << "Top brigthness: " << topBrig << std::endl;
    std::cout << "Middle brigthness: " << middleBrig << std::endl;
    std::cout << "Bottom brigthness: " << bottomBrig << std::endl;

    std::cout << "red coverage: " << hsvTop.maskCoverage << std::endl;
    std::cout << "orange coverage: " << hsvMiddle.maskCoverage << std::endl;
    std::cout << "green coverage: " << hsvBottom.maskCoverage << std::endl;

    std::cout << "GrayScale test result:    " << translate_TL_state(grayScaleTest) << std::endl;
    std::cout << "HSV test result:          " << translate_TL_state(hsvTest) << std::endl;
    if (grayScaleTest != hsvTest)
        std::cout << "*****NOT EQUAL RESULTS*****" << std::endl;
    std::cout << "==============================================" << std::endl;




    //auto finish = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> timeElapsed = finish - start;
    //std::cout << "State found after: " << timeElapsed.count() << " ms." << std::endl;

    if (all_zero({hsvTop.maskCoverage, hsvMiddle.maskCoverage, hsvBottom.maskCoverage}))
    {
        return Inactive;
    }
    else
    {
        return grayScaleTest;
    }
}




void save_found_crop(cv::Mat & mat, dlib::mmod_rect rectangle, int index)
{
    cv::Mat cropped = crop_image(mat, rectangle);

    cvtColor(cropped, cropped, CV_BGR2RGB);
    std::cout << "Saving: crop_" << std::to_string(index) << ".png" << std::endl;
    cv::imwrite("crops/crop_" + std::to_string(index) + ".png" , cropped);
}

cv::Mat crop_image(cv::Mat & mat, dlib::mmod_rect cropRectangle)
{
    cv::Rect roi(cropRectangle.rect.left(), cropRectangle.rect.top(), cropRectangle.rect.width(), cropRectangle.rect.height());
    cv::Mat cropped = mat(roi);

    return cropped;
}





















