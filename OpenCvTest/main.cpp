#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void* );

/** @function main */
int main( int argc, char** argv )
{
    /// Load source image and convert it to gray
    src = imread( argv[1], 1 );

    /// Convert image to gray and blur it
    cvtColor( src, src_gray, CV_BGR2GRAY );
    //blur( src_gray, src_gray, Size(3,3) );

    /// Create Window
    char* source_window = "Source";
    namedWindow( source_window, 0 );
    imshow( source_window, src );

    createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );

    waitKey(0);
    return(0);
}


long best_contour(vector<vector<Point>> &contours)
{
    long best = 0;
    long best_id = 0;
    int id = -1;
    for (std::vector<Point> & c : contours)
    {
        ++id;
        long size = c.size();
        if (size > best)
        {
            best_id = id;
            best = size;
        }
    }

    //std::cout << "Best: " << best << std::endl;
    return best_id;
}

bool contains(std::vector<Point> & contour, long x, long y )
{
    for (Point & p : contour)
    {
        if (p.x == x && p.y == y)
            return true;
    }
    
    return false;
}

void fill_cont(Mat &mat, Mat & cont)
{
    for (int r = 0; r < mat.rows; ++r) {
        bool draw = true;
        int leftB = -1;
        int rightB = -1;

        for (int c = 0; c < mat.cols; ++c) {
            if (cont.at<uchar>(r,c) == 255) {
                if (c < mat.cols / 2)
                    leftB = c;
                else
                    rightB = c;
            }
        }

        if (leftB == -1 && rightB == -1)
        {
            continue;
        }
        if (leftB != -1 && rightB == -1)
        {
            for (int c = leftB; c < mat.cols; ++c)
                mat.at<uchar>(r,c) = 255;

            continue;
        }
        if (leftB == -1 && rightB != -1)
        {
            for (int c = 0; c < rightB; ++c)
                mat.at<uchar>(r,c) = 255;

            continue;
        }
        else
        {
            for (int c = leftB; c < rightB; ++c)
                mat.at<uchar>(r,c) = 255;
        }

        /*
        for (int c = 0; c < mat.cols; ++c) {


            if (cont.at<uchar>(r,c) == 255)
            {
                //first border hit
                if (!draw)
                    draw = true;
                else
                    draw = false;
            }

            if (draw) {
                mat.at<uchar>(r,c) = 255;
            }


        }
        */
    }
}


/** @function thresh_callback */
void thresh_callback(int, void* )
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    //findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( canny_output, contours, hierarchy, RETR_EXTERNAL  , CV_CHAIN_APPROX_SIMPLE , Point(0, 0) );


    /// Draw contours
    //Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    Mat1b drawing = Mat::zeros( canny_output.size(), CV_8UC1 );


    long contId = best_contour(contours);

    //does not work
    //cout << "Contour is convex: " << isContourConvex(contours.at(contId)) << endl;
    drawContours(drawing, contours, contId, Scalar::all(255));//CV_FILLED

    Mat1b filled = Mat::zeros(drawing.size(),CV_8UC1);
    fill_cont(filled, drawing);

    for( int i = 0; i< contours.size(); i++ )
    {
        //drawContours( drawing, contours, i, Scalar::all(255), 1, 8, noArray(), 0, Point() );
        //fillPoly(drawing,contours.at(i), Scalar::all(255));

    }

    /// Show in a window
    namedWindow( "Contours", 0 );
    imshow( "Contours", drawing );

    namedWindow( "fill", 0 );
    imshow( "fill", filled );

    //namedWindow( "Canny", 0 );
    //imshow( "Canny", canny_output );
}

/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo( int, void* );


int main( int argc, char** argv )
{
    /// Load an image
    src = imread( argv[1], 1 );

    /// Convert the image to Gray
    cvtColor( src, src_gray, CV_RGB2GRAY );

    /// Create a window to display results
    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    /// Create Trackbar to choose type of Threshold
    createTrackbar( trackbar_type,
                    window_name, &threshold_type,
                    max_type, Threshold_Demo );

    createTrackbar( trackbar_value,
                    window_name, &threshold_value,
                    max_value, Threshold_Demo );

    /// Call the function to initialize
    Threshold_Demo( 0, 0 );

    /// Wait until user finishes program
    while(true)
    {
        int c;
        c = waitKey( 20 );
        if( (char)c == 27 )
        { break; }
    }

}

void Threshold_Demo( int, void* )
{


    threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );

    imshow( window_name, dst );
}
*/