#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>

using namespace cv;
using namespace std;

/// Global variables
int threshold_value = 90;
int threshold_type = 1;
int const max_value = 255;
int const max_BINARY_value = 255;

Mat src, gray, hsv;
char* window_name = "Threshold Demo";

char* trackbar_value = "Value";
bool maskOut = false;
bool save = false;
std::string imgFile;


long best_contour(vector<vector<Point>> &contours, vector<Vec4i>& hierarchy)
{
    long best = 0;
    long best_id = 0;
    int id = -1;
    for (long i = 0; i < contours.size() ; ++i)
    {
        std::vector<Point> & c = contours.at(i);
        ++id;
        long size = c.size();
        if (hierarchy.at(i)[2] < 0 && size > best)
        {
            best_id = id;
            best = size;
        }

        if (hierarchy.at(i)[2] > 0)
            std::cout << "Has child" << std::endl;
    }

    //std::cout << "Best: " << best << std::endl;
    return best_id;
}

/// Function headers
void Threshold_Demo( int, void* );


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

float get_total_brightness_in_range(cv::Mat &img, int lowRow, int highRow, int lowCol, int highCol, int &partPixelCount, int& max)
{
    float brightness = 0.0f;
    partPixelCount = 0;

    for (int row = lowRow; row < highRow; ++row)
    {
        for (int col = lowCol; col < highCol; ++col) {
            uchar px = img.at<uchar>(row, col);
            brightness += px;


            ++partPixelCount;
        }
    }

    return brightness;
}


bool should_mask_contour(cv::Mat & grayImg, int& max_val)
{
    float bordersBrigthness  = 0.0f;

    int rowCount = (int)ceil(grayImg.rows * 0.06f);
    int columnCount = (int)ceil(grayImg.cols * 0.15f);
    int leftPx, rightPx, topPx, bottomPx, centerPx;
    int leftMax, rightMax, topMax, bottomMax, dummy;

    float leftBorderBrightness = get_total_brightness_in_range(grayImg, rowCount, grayImg.rows - rowCount, 0,
                                                               columnCount, leftPx, leftMax);
    float rightBorderBrightness = get_total_brightness_in_range(grayImg, rowCount, grayImg.rows - rowCount,
                                                                grayImg.cols - columnCount, grayImg.cols, rightPx, rightMax);
    float topBorderBrigthness = get_total_brightness_in_range(grayImg, 0, rowCount, 0, columnCount, topPx, topMax);
    float bottomBorderBrightness = get_total_brightness_in_range(grayImg, grayImg.rows - rowCount, grayImg.rows, 0,
                                                                 columnCount, bottomPx, bottomMax);

    //max_val = max({leftMax, rightMax, topMax, bottomMax});
    bordersBrigthness = (
                                leftBorderBrightness +
                                rightBorderBrightness +
                                topBorderBrigthness  +
                                bottomBorderBrightness
                        ) / ((float)(leftPx + rightPx + topPx + bottomPx));

    float centerBrightness = get_total_brightness_in_range(grayImg, rowCount, grayImg.rows - rowCount, columnCount,
                                                           grayImg.cols - columnCount, centerPx, dummy);

    centerBrightness = centerBrightness / (float)centerPx;
/*
    cout << "------------" << endl;
    cout << "Center value: " << centerBrightness << endl;
    cout << "Border value: " << bordersBrigthness << endl;
    cout << "------------" << endl;
*/
    return (bordersBrigthness > centerBrightness);
}


void remove_background(cv::Mat & grayImg, cv::Mat & hsvImg, std::pair<int, int> & verticalBoundaries, bool maskOut, int threshold_type)
{
    using namespace cv;
    using namespace std;

    if (maskOut)
    {

        Mat1b contour, maskedImage;
        Mat hsvMasked, thresholdOut;

        threshold(grayImg, thresholdOut, threshold_value, 255, threshold_type); //130

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        contour = Mat::zeros(grayImg.size(), CV_8UC1 );
        maskedImage = Mat::zeros(grayImg.size(), CV_8UC1);
        hsvMasked = Mat::zeros(hsvImg.size(), CV_8UC3);

        findContours(thresholdOut, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE , Point(0, 0));

        long contId = best_contour(contours, hierarchy);
        drawContours(contour, contours, contId, Scalar::all(255), CV_FILLED);



        vector<Vec3f> circles;
        HoughCircles(contour, circles, HOUGH_GRADIENT, 2, contour.rows/8, 200, 15, 0, (float)contour.cols * 0.4f);

        //cout << "Circle count: " << circles.size() << endl;

        if (circles.size() > 0)
            cout << circles.size() <<  " circles found for image " << imgFile << endl;

        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            radius *= 1.2f;

            circle(contour, center, radius, Scalar::all(255), CV_FILLED, 8, 0);

        }
/*
        namedWindow("contour", 0);
        imshow("contour", contour);
*/


        std::pair<int, int> tmp = find_vertical_boundaries(contour);
        verticalBoundaries.first = tmp.first;
        verticalBoundaries.second = tmp.second;

        grayImg.copyTo(maskedImage, contour);
        hsvImg.copyTo(hsvMasked, contour);

        grayImg = maskedImage;
        hsvImg = hsvMasked;


    }
    else
    {
        verticalBoundaries.first = 0;
        verticalBoundaries.second = grayImg.rows;
        //cout << "No masking" << endl;
    }

}

void video(string videoFile)
{
    VideoCapture capture(videoFile);
    Mat frame;

    namedWindow(videoFile, 0);
    while (true)
    {
        capture >> frame;
        if (frame.empty())
            break;



        imshow(videoFile, frame);
        waitKey(67);
    }

    waitKey(0);
}

int main( int argc, char** argv )
{
    /// Load an image
    imgFile = (std::string)argv[1];
    src = imread( imgFile );
    string saveArg = (std::string)argv[2];
    save = saveArg == "-s";

    if (saveArg == "-v")
    {
        video(imgFile);
        return 0;
    }


    Mat g;
    cvtColor(src, g, CV_BGR2GRAY);
    //normalize(g, g, 0, 95, NORM_MINMAX);
    int max;

    maskOut = should_mask_contour(g, max);

    //cout << "max: " << max << endl;

    /// Create a window to display results
    namedWindow( window_name, 0 );

    createTrackbar( trackbar_value,
                    window_name, &threshold_value,
                    max_value, Threshold_Demo );

    /// Call the function to initialize
    Threshold_Demo( 0, 0 );

    /// Wait until user finishes program
    while(true && !save)
    {
        int c;
        c = waitKey( 20 );
        if( (char)c == 27 )
        { break; }
    }

}


void Threshold_Demo( int, void* )
{
    cvtColor(src, gray, CV_BGR2GRAY);
    cvtColor(src, hsv, CV_BGR2HSV);

    Mat thresh_otsu, thresh_triangle;
    thresh_otsu = Mat(gray);
    //thresh_triangle = Mat(gray);
    std::pair<int, int> verticalBoundaries;

//    remove_background(gray, hsv, verticalBoundaries, maskOut, CV_THRESH_BINARY_INV);
    remove_background(thresh_otsu, hsv, verticalBoundaries, maskOut, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
    //remove_background(thresh_triangle, hsv, verticalBoundaries, maskOut, CV_THRESH_BINARY_INV + CV_THRESH_TRIANGLE);

    if ( save && maskOut)
    {

        string otsuFile = "OTSU/otsu_" + imgFile;
        string triangeFile = "TRIANGLE/triangle_" + imgFile;

        imwrite(otsuFile, thresh_otsu);
        cout << "Saving " << otsuFile << endl;
//        imwrite(triangeFile, thresh_triangle);
//        cout << "Saving " << triangeFile << endl;

        return;
    }
    else if ( save )
        return;


    normalize(gray, gray, 0, 95, NORM_MINMAX); // 150

    Mat topPartGray, middlePartGray, bottomPartGray;
    int partHeight = (verticalBoundaries.second - verticalBoundaries.first) / 3;

    topPartGray =       Mat(thresh_otsu, Rect(0, verticalBoundaries.first, src.cols, partHeight));
    middlePartGray =    Mat(thresh_otsu, Rect(0, verticalBoundaries.first + partHeight, src.cols, partHeight));
    bottomPartGray =    Mat(thresh_otsu, Rect(0, verticalBoundaries.first + 2 * partHeight, src.cols, partHeight));
    /*******************************************************/

    imshow(window_name, thresh_otsu);

    ///namedWindow("OTSU", 0);
    //imshow("OTSU", thresh_otsu);

    //namedWindow("triangle", 0);
    //imshow("triangle", thresh_triangle);
/*
    namedWindow("Top", 0);
    imshow("Top", topPartGray);

    namedWindow("Middle", 0);
    imshow("Middle", middlePartGray);

    namedWindow("Bottom", 0);
    imshow("Bottom", bottomPartGray);
*/

}
