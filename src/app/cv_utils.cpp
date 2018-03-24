#include <dlib/image_transforms/interpolation.h>
#include "cv_utils.h"


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

bool all_below_or_equal(std::vector<float> values, float threshold)
{
    for (float val : values)
    {
        if (val > threshold)
        {
            return false;
        }
    }
    return true;
}

bool all_over_or_equal(std::vector<float> values, float threshold)
{
    for (float val : values)
    {
        if (val < threshold)
        {
            return false;
        }
    }
    return true;
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
        case Ambiguous:
            return "Ambiguous";
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


void fix_mask(cv::Mat1b &mask)
{
    int fromCol = (int)((mask.cols / 2.0f) - (mask.cols * 0.25f));
    int toCol = (int)((mask.cols / 2.0f) + (mask.cols * 0.25f));
    int fromRow = (int)(mask.rows * 0.10f);
    int toRow = (int)(mask.rows * 0.90f);

    bool shouldFix = false;
    uchar px;
    for (int r = fromRow; r < toRow; ++r)
    {
        for (int c = fromCol; c < toCol; ++c)
        {
            px = mask.at<uchar>(r, c);
            if (px == 0)
            {
                shouldFix = true;
                break;
            }
        }
        if (shouldFix)
        {
            break;
        }
    }

    if (shouldFix)
    {
        fromCol = (int)((mask.cols / 2.0f) - (mask.cols * 0.35f));
        toCol = (int)((mask.cols / 2.0f) + (mask.cols * 0.35f));
        fromRow = (int)(mask.rows * 0.05f);
        toRow = (int)(mask.rows * 0.95f);

        cv::Mat1b fixMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        for (int r = fromRow; r < toRow; ++r)
        {
            for (int c = fromCol; c < toCol; ++c)
            {
                fixMask.at<uchar>(r,c) = 255;
            }
        }

        mask = mask | fixMask;
    }
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
        pxInRow = 0;
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

    for (int r = img.rows - 1 ; r > img.rows / 2; --r) {
        pxInRow = 0;
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

float get_sum_in_range(cv::Mat &img, int lowRow, int highRow, int lowCol, int highCol, int &pixelCount)
{
    float brightness = 0.0f;
    pixelCount = 0;

    for (int row = lowRow; row < highRow; ++row)
    {
        for (int col = lowCol; col < highCol; ++col) {
            brightness += img.at<uchar>(row, col);
            ++pixelCount;
        }
    }

    return brightness;
}

bool should_mask_contour(cv::Mat & grayImg)
{
    float bordersBrigthness  = 0.0f;

    int rowCount = (int)ceil(grayImg.rows * 0.06f);
    int columnCount = (int)ceil(grayImg.cols * 0.15f);
    int leftPxs, rightPxs, topPxs, bottomPxs, centerPxs;

    float leftBorderBrightness = get_sum_in_range(grayImg, rowCount, grayImg.rows - rowCount, 0, columnCount, leftPxs);
    float rightBorderBrightness = get_sum_in_range(grayImg, rowCount, grayImg.rows - rowCount, grayImg.cols - columnCount, grayImg.cols, rightPxs);
    float topBorderBrigthness = get_sum_in_range(grayImg, 0, rowCount, 0, columnCount, topPxs);
    float bottomBorderBrightness = get_sum_in_range(grayImg, grayImg.rows - rowCount, grayImg.rows, 0, columnCount, bottomPxs);

    bordersBrigthness = (leftBorderBrightness + rightBorderBrightness + topBorderBrigthness + bottomBorderBrightness) /
                        ((float)(leftPxs + rightPxs + topPxs + bottomPxs));

    float centerBrightness = get_sum_in_range(grayImg, rowCount, grayImg.rows - rowCount, columnCount, grayImg.cols - columnCount, centerPxs);
    centerBrightness = centerBrightness / (float)centerPxs;

    return (bordersBrigthness > centerBrightness);
}


void remove_background(cv::Mat & grayImg, cv::Mat & hsvImg, bool verbose)
{
    using namespace cv;
    using namespace std;

    if (should_mask_contour(grayImg))
    {
        Mat1b contour, maskedImage;
        Mat hsvMasked, thresholdOut;

        //OTSU thresholding, so threshold value does not make any difference.
        threshold(grayImg, thresholdOut, 126, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU); //130

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        contour = Mat::zeros(grayImg.size(), CV_8UC1);

        findContours(thresholdOut, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        long contId = best_contour(contours, hierarchy);
        drawContours(contour, contours, contId, Scalar::all(255), CV_FILLED);

        std::pair<int, int> verticalBoundaries = find_vertical_boundaries(contour);

        Rect imgRoi(0, verticalBoundaries.first, grayImg.cols, verticalBoundaries.second - verticalBoundaries.first);
        Mat1b croppedMask(contour, imgRoi);
        fix_mask(croppedMask);

        maskedImage = Mat::zeros(croppedMask.size(), CV_8UC1);
        hsvMasked = Mat::zeros(croppedMask.size(), CV_8UC3);

        grayImg = Mat(grayImg, imgRoi);
        hsvImg = Mat(hsvImg, imgRoi);

        grayImg.copyTo(maskedImage, croppedMask);
        hsvImg.copyTo(hsvMasked, croppedMask);

        grayImg = maskedImage;
        hsvImg = hsvMasked;
    }
}

bool clear_hsv_test(float top, float middle, float bottom)
{
    float threshold = 0.1f;
    if (top < threshold && middle < threshold && bottom > threshold)
        return true;
    if (top < threshold && middle > threshold && bottom < threshold)
        return true;
    if (top > threshold && middle < threshold && bottom < threshold)
        return true;

    return false;
}

TLState get_traffic_light_state2(dlib::matrix<dlib::rgb_pixel> dlibImg, bool verbose)
{
    //DLIB IMAGE IS IN RGB!

    cv::Mat dlibMatImg = dlib::toMat(dlibImg);
    return get_traffic_light_state(dlibMatImg, verbose);
}

bool is_orange(float grayTop, float grayMiddle, float grayBottom, float hsvTop, float hsvMiddle, float hsvBottom)
{
    if (all_below_or_equal({hsvTop, hsvMiddle, hsvBottom}, 0))
        return false;

    if (!all_over_or_equal({hsvTop, hsvBottom}, hsvMiddle))
        return false;

    if (all_below_or_equal({grayTop, grayBottom}, grayMiddle * 0.9f))
        return true;

    return false;
}

TLState get_traffic_light_state(cv::Mat & img, bool verbose)
{
    using namespace cv;

    Stopwatch s;
    s.start();

    Mat gray, hsv;
    //image from dlib is rgb not bgr.
    cvtColor(img, gray, CV_RGB2GRAY);
    cvtColor(img, hsv, CV_RGB2HSV);

    //remove_background(gray, hsv, verbose);

    normalize(gray, gray, 0, 95, NORM_MINMAX); // 150

    Mat topPartGray, middlePartGray, bottomPartGray, topPartHsv, middlePartHsv, bottomPartHsv;
    int partHeight = (gray.rows) / 3;

    topPartGray =       Mat(gray, Rect(0, 0, img.cols, partHeight));
    middlePartGray =    Mat(gray, Rect(0, partHeight, img.cols, partHeight));
    bottomPartGray =    Mat(gray, Rect(0, 2 * partHeight, img.cols, partHeight));

    topPartHsv =       Mat(hsv, Rect(0, 0, img.cols, partHeight));
    middlePartHsv =    Mat(hsv, Rect(0, partHeight, img.cols, partHeight));
    bottomPartHsv =    Mat(hsv, Rect(0, 2 * partHeight, img.cols, partHeight));

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

    if (verbose)
    {
        s.stop();
        std::cout << "Time: " << s.elapsed_milliseconds() << " ms" << std::endl;

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


    if (all_below_or_equal({hsvTop.maskCoverage, hsvMiddle.maskCoverage, hsvBottom.maskCoverage}, 0))
    {
        return Inactive;
    }

    if (hsvTest == grayScaleTest)
    {
        return grayScaleTest;
    }

    if (is_orange(topBrig, middleBrig, bottomBrig, hsvTop.maskCoverage, hsvMiddle.maskCoverage, hsvTop.maskCoverage))
    {
        return Orange;
    }

    if (clear_hsv_test(hsvTop.maskCoverage, hsvMiddle.maskCoverage, hsvBottom.maskCoverage))
    {
        return hsvTest;
    }

    return grayScaleTest;

}

void save_found_crop(cv::Mat & mat, dlib::mmod_rect rectangle, int imgIndex, int labelIndex)
{
    cv::Mat cropped = crop_image(mat, rectangle);

    cvtColor(cropped, cropped, CV_BGR2RGB);

    std::string fn = "crops/crop_" + std::to_string(imgIndex) + "_" + std::to_string(labelIndex) + ".png";

    std::cout << "Saving: " << fn << std::endl;
    cv::imwrite(fn , cropped);
}

cv::Mat crop_image(const cv::Mat & mat, const dlib::rectangle& cropRectangle)
{
    cv::Rect roi(cropRectangle.left(), cropRectangle.top(), cropRectangle.width(), cropRectangle.height());
    cv::Mat cropped = mat(roi);

    return cropped;
}

dlib::matrix<dlib::rgb_pixel> crop_image(const dlib::matrix<dlib::rgb_pixel> &original, const dlib::rectangle& cropRectangle, bool exactCrop)
{
    dlib::chip_details chipDetail;
   /* if (exactCrop)
    {
        if (cropRectangle.width() < 6 || cropRectangle.height() < 6)
            throw new std::runtime_error("Cannot size down that small rectangle!");

        chipDetail = dlib::chip_details(dlib::rectangle(cropRectangle.left(), cropRectangle.top(), cropRectangle.right() - 5, cropRectangle.bottom() - 5));
    }

    else
	*/
    chipDetail = dlib::chip_details(cropRectangle);

    dlib::matrix<dlib::rgb_pixel> crop;

    dlib::extract_image_chip(original, chipDetail , crop);

    return crop;
}

dlib::rgb_pixel get_color_for_state(TLState state)
{
    std::cout << "Detected state: " << state << std::endl;
    if (state == Red)
        return dlib::rgb_pixel(255, 0, 0);

    if (state == Green)
        return dlib::rgb_pixel(0, 255, 0);

    if (state == Orange || state == RedOrange)
        return dlib::rgb_pixel(255, 158, 48);

    return dlib::rgb_pixel(10, 10, 10);
}

dlib::rectangle resized_rectangle(dlib::rectangle original, dlib::rectangle sizeRectangle)
{
    long halfDeltaW = (sizeRectangle.width() - original.width()) / 2;
    long halfDeltaH = (sizeRectangle.height() - original.height()) / 2;

    dlib::rectangle sized(  original.left() - halfDeltaW,   original.top() - halfDeltaH,
                            original.right() + halfDeltaW,  original.bottom() + halfDeltaH);

    if (sized.width() != sizeRectangle.width())
    {
        sized.set_right(sized.right() + (sizeRectangle.width() - sized.width()));
    }

    if (sized.height() != sizeRectangle.height())
    {
        sized.set_bottom(sized.bottom() + (sizeRectangle.height() - sized.height()));
    }
    
    
    return sized;

}

void save_found_crop(cv::Mat &mat, dlib::mmod_rect detRect, std::string fileName, dlib::rectangle sizeRect)
{
	using namespace cv;
    cv::Mat cropped = crop_image(mat, detRect.rect);
    cv::cvtColor(cropped, cropped, CV_BGR2RGB);
    if (sizeRect.is_empty())
    {
        //cv::Mat cropped = crop_image(mat, detRect.rect);
        std::cout << "Saving: " << fileName << std::endl;
        cv::imwrite(fileName, cropped);
        return;
    }
    else
    {
        if (sizeRect.width() > detRect.rect.width() && sizeRect.height() > detRect.rect.height())
        {	
            long DeltaW = sizeRect.width() - detRect.rect.width();
            long DeltaH = sizeRect.height() - detRect.rect.height();

            cv::Mat resized = cv::Mat::zeros(sizeRect.height(), sizeRect.width(), mat.type());
	    //cv::Scalar black = cv::Scalar::all(0);
            //cv::copyMakeBorder(cropped, resized, 0, DeltaH, 0, DeltaW, BORDER_CONSTANT, black);
	    cropped.copyTo(resized(cv::Rect(0, 0, cropped.cols, cropped.rows)));
            std::cout << "Saving sized up: " << fileName << std::endl;
            cv::imwrite(fileName, resized);
        }
        else
        {
            cv::Mat resized;
            cv::resize(cropped, resized, cv::Size(sizeRect.width(), sizeRect.height()));
            std::cout << "Saving sized down: " << fileName << std::endl;
            cv::imwrite(fileName, resized);
        }
    }
}

void convert_to_grayscale(dlib::matrix<dlib::rgb_pixel>& image)
{
    int lumma;
    for (long row = 0; row < image.nr(); ++row)
    {
        for (long col = 0; col < image.nc(); ++col)
        {
            dlib::rgb_pixel px = image(row, col);
            lumma = 0.299 * px.red + 0.587 * px.green + 0.114 * px.blue;
            px.red = lumma; px.green = lumma; px.blue = lumma;
            dlib::assign_pixel(image(row, col), px);
        }
    }
}

void convert_to_grayscale(dlib::array2d<dlib::rgb_pixel>& image)
{
    int lumma;
    for (long row = 0; row < image.nr(); ++row)
    {
        for (long col = 0; col < image.nc(); ++col)
        {
            dlib::rgb_pixel px = image[row][col];
            lumma = 0.299 * px.red + 0.587 * px.green + 0.114 * px.blue;
            px.red = lumma; px.green = lumma; px.blue = lumma;
            dlib::assign_pixel(image[row][col], px);
        }
    }
}


dlib::rectangle transform_rectangle_back(const dlib::rectangle& rect, const float scaleFactor)
{
	float scaleBackFactor = 1.0f / scaleFactor;
	long newLeft = rect.tl_corner().x() * scaleBackFactor;
	long newTop = rect.tl_corner().y() * scaleBackFactor;
	long newWidth = rect.width() * scaleBackFactor;
	long newHeight = rect.height() * scaleBackFactor;

	return dlib::rectangle(newLeft, newTop, newLeft + newWidth, newTop + newHeight);
}

