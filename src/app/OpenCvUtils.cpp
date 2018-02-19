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

std::vector<std::pair<cv::Scalar, cv::Scalar>> whiteBounds = {
        std::pair<cv::Scalar, cv::Scalar>(lowerWhite, upperWhite)
};

int thread_count = 6;
dlib::mutex count_mutex;
dlib::signaler count_signaler(count_mutex);

void show(cv::Mat & img, std::string winName = "")
{
    if (winName == "")
        winName = "Window";

    cv::namedWindow(winName, 0);
    cv::imshow(winName, img);
    cv::waitKey(0);
}

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

long best_contour(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i>& hierarchy)
{
    long bestSize = 0;
    long best_id = 0;
    int id = -1;
    for (long i = 0; i < contours.size() ; ++i)
    {
        std::vector<cv::Point> & c = contours.at(i);
        ++id;
        long size = c.size();
        if (hierarchy.at(i)[2] < 0 && size > bestSize)
        {
            best_id = id;
            bestSize = size;
        }

        if (hierarchy.at(i)[2] > 0)
            std::cout << "Has child" << std::endl;
    }

    //std::cout << "Best: " << bestSize << std::endl;
    return best_id;
}

TLState get_most_possible_state(std::vector<std::pair<TLState, float>> states)
{
    TLState mostPossibleState = Inactive;
    float highestBrightness = 0.0f;

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

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            brightness += img.at<uchar>(col, row);
        }
    }

    brightness = brightness / (float)(img.rows * img.cols);

    return brightness;

}

float get_mask_coverage(HsvTestParam * partInfo)
{
    long whitePixelCount = 0;

    for (int row = 0; row < partInfo->resultMask.rows; ++row) {
        for (int col = 0; col < partInfo->resultMask.cols; ++col) {
            if (partInfo->resultMask.at<uchar>(row, col) == 255)
            {
                ++whitePixelCount;
            }

        }
    }

    long totalPixelCount = partInfo->resultMask.rows * partInfo->resultMask.cols;
    float coveragePerc = ((float)(whitePixelCount)) / ((float)(totalPixelCount));

    return coveragePerc;
}

void hsvTest(void *param)
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

#ifdef THREADING
    dlib::auto_mutex countLock(count_mutex);
    --thread_count;
    count_signaler.signal();
#endif
}

void grayScaleTest(void * param)
{
    GrayScaleTestParam * info = (GrayScaleTestParam*)param;
    info->result = get_average_brightness(info->imgPart);

#ifdef THREADING
    dlib::auto_mutex countLock(count_mutex);
    --thread_count;
    count_signaler.signal();
#endif

}

std::pair<int, int> find_vertical_boundaries(cv::Mat1b & img)
{
    int top = -1;
    int bottom = -1;
    int pxInRow = 0;
    float pxInRowThreshold = 0.65f;
    for (int r = 0; r < img.rows / 2; ++r) {

        for (int c = 0; c < img.cols; ++c) {
            if (img.at<uchar>(r,c) == 255)
            {
                ++pxInRow;
            }
        }
        if (((float)pxInRow / (float)img.cols) > pxInRowThreshold)
        {
            top = r;
            break;
        }
    }

    pxInRow = 0;
    for (int r = img.rows - 1 ; r > img.rows / 2; --r) {
        for (int c = 0; c < img.cols; ++c) {
            if (img.at<uchar>(r,c) == 255)
            {
                ++pxInRow;
            }
        }
        if (((float)pxInRow / (float)img.cols) > pxInRowThreshold)
        {
            bottom = r;
            break;
        }
    }

    top = top == -1 ? 0 : top;
    bottom = bottom == -1 ? img.rows : bottom;

    return std::make_pair(top, bottom);
};

float get_brigthness_in_range(cv::Mat & img, int lowRow, int highRow, int lowCol, int highCol)
{
    float brightness = 0.0f;
    int pixelCount = 0;

    for (int row = lowRow; row < highRow; ++row)
    {
        for (int col = lowCol; col < highCol; ++col) {
            brightness += img.at<uchar>(row, col);
            ++pixelCount;
        }
    }

    return (brightness / (float)(pixelCount));
}

bool should_mask_contour(cv::Mat & grayImg)
{
    float bordersBrigthness  = 0.0f;

    int rowCount = (int)ceil(grayImg.rows * 0.06f);
    int columnCount = (int)ceil(grayImg.cols * 0.15f);

    float leftBorderBrightness = get_brigthness_in_range(grayImg, rowCount, grayImg.rows - rowCount, 0, columnCount);
    float rightBorderBrightness = get_brigthness_in_range(grayImg, rowCount, grayImg.rows - rowCount, grayImg.cols - columnCount, grayImg.cols);
    float topBorderBrigthness = get_brigthness_in_range(grayImg, 0, rowCount, 0, columnCount);
    float bottomBorderBrightness = get_brigthness_in_range(grayImg, grayImg.rows - rowCount, grayImg.rows, 0, columnCount);

    bordersBrigthness = (leftBorderBrightness + rightBorderBrightness + topBorderBrigthness + bottomBorderBrightness) / 4.0f;

    float centerBrightness = get_brigthness_in_range(grayImg, rowCount, grayImg.rows - rowCount, columnCount, grayImg.cols - columnCount);

    return (bordersBrigthness > centerBrightness);
}


void remove_background(cv::Mat & grayImg, cv::Mat & hsvImg, std::pair<int, int> & verticalBoundaries, bool verbose)
{
    using namespace cv;
    using namespace std;

    if (should_mask_contour(grayImg))
    {
        cout << "Doing contour masking" << endl;

        Mat1b contour, maskedImage;
        Mat hsvMasked, thresholdOut;

        threshold(grayImg, thresholdOut, 126, 255, CV_THRESH_BINARY_INV); //130

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        contour = Mat::zeros(grayImg.size(), CV_8UC1 );
        maskedImage = Mat::zeros(grayImg.size(), CV_8UC1);
        hsvMasked = Mat::zeros(hsvImg.size(), CV_8UC3);

        findContours(thresholdOut, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE , Point(0, 0));


        long contId = best_contour(contours, hierarchy);
        drawContours(contour, contours, contId, Scalar::all(255), CV_FILLED);

        std::pair<int, int> tmp = find_vertical_boundaries(contour);
        verticalBoundaries.first = tmp.first;
        verticalBoundaries.second = tmp.second;

        grayImg.copyTo(maskedImage, contour);
        hsvImg.copyTo(hsvMasked, contour);

        grayImg = maskedImage;
        hsvImg = hsvMasked;

        if (verbose)
        {
            show(contour, "Contour");
            show(maskedImage, "Masked image");
        }


    }

    verticalBoundaries.first = 0;
    verticalBoundaries.second = grayImg.rows;
}

bool clear_hsv_test(float top, float middle, float bottom)
{
    float threshold = 0.1f;
    if (top < threshold && middle < threshold && bottom > 0)
        return true;
    if (top < threshold && middle > 0 && bottom < threshold)
        return true;
    if (top > 0 && middle < threshold && bottom < threshold)
        return true;

    return false;
}

TLState get_traffic_light_state(cv::Mat & img, bool verbose)
{
    using namespace cv;
    Logger log("StateLogger.txt");

    Stopwatch s;
    s.start();

    Mat gray, hsv;
    cvtColor(img, gray, CV_BGR2GRAY);
    cvtColor(img, hsv, CV_BGR2HSV);

    std::pair<int, int> verticalBoundaries;
    remove_background(gray, hsv, verticalBoundaries, verbose);

    normalize(gray, gray, 0, 95, NORM_MINMAX); // 150

    Mat topPartGray, middlePartGray, bottomPartGray, topPartHsv, middlePartHsv, bottomPartHsv;
    int partHeight = (verticalBoundaries.second - verticalBoundaries.first) / 3;

    topPartGray =       Mat(gray, Rect(0, verticalBoundaries.first, img.cols, partHeight));
    middlePartGray =    Mat(gray, Rect(0, verticalBoundaries.first + partHeight, img.cols, partHeight));
    bottomPartGray =    Mat(gray, Rect(0, verticalBoundaries.first + 2 * partHeight, img.cols, partHeight));

    topPartHsv =       Mat(hsv, Rect(0, verticalBoundaries.first, img.cols, partHeight));
    middlePartHsv =    Mat(hsv, Rect(0, verticalBoundaries.first + partHeight, img.cols, partHeight));
    bottomPartHsv =    Mat(hsv, Rect(0, verticalBoundaries.first + 2 * partHeight, img.cols, partHeight));

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
    hsvTest(&hsvTop);
    hsvTest(&hsvMiddle);
    hsvTest(&hsvBottom);

    grayScaleTest(&tp);
    grayScaleTest(&mp);
    grayScaleTest(&bp);
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


    log.write_line(std::to_string(topBrig));
    
    if (verbose)
    {
        s.stop();
        std::cout << "Time: " << s.elapsed() << " ms" << std::endl;

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
    }


    if (all_zero({hsvTop.maskCoverage, hsvMiddle.maskCoverage, hsvBottom.maskCoverage}))
    {
        return Inactive;
    }

    if (hsvTest == grayScaleTest)
    {
        return grayScaleTest;
    }

    if (clear_hsv_test(hsvTop.maskCoverage, hsvMiddle.maskCoverage, hsvBottom.maskCoverage))
    {
        return hsvTest;
    }

    return grayScaleTest;

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

void append_to_file(std::string fileName, std::string message)
{
    using namespace std;

    ofstream file;
    file.open(fileName, ios::out | ios::app);

    file << message << endl;

    file.close();


}





















