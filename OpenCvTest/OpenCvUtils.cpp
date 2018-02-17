//
// Created by azgra on 10.2.18.
//

#include "OpenCvUtils.h"

Color get_average_color(cv::Mat & img)
{
    long pixels = img.cols * img.rows;

    cv::Scalar s = cv::sum(img);
    Color averageColor;

    averageColor.blue =     (int)(s.val[0] / pixels);
    averageColor.green =    (int)(s.val[1] / pixels);
    averageColor.red =      (int)(s.val[2] / pixels);

    return  averageColor;
}

float get_average_brightness(cv::Mat & img)
{
    Color c = get_average_color(img);

    return (float)(c.combined()) / (float)(765);
}

TLState get_traffic_light_state(cv::Mat & img)
{
    using namespace cv;

    //It is best probably to mask white, brightest spots
    Scalar lowerWhite = Scalar(0,0,225);
    Scalar upperWhite = Scalar(255,255,255);

    Scalar lower_red_0 = Scalar(0, 100, 100);
    Scalar upper_red_0 = Scalar(15, 255, 255);
    Scalar lower_red_1 = Scalar(180 - 15, 100, 100);
    Scalar upper_red_1 = Scalar(180, 255, 255);

    Scalar lowerGreen = Scalar(44, 71, 82);
    Scalar upperGreen = Scalar(105, 255, 255);


    Mat topPart, middlePart, bottomPart;
    Mat1b topMask, middleMask, bottomMask;

    int partHeight = img.rows / 3;

    topPart =       Mat(img, Rect(0,0,img.cols, partHeight));
    middlePart =    Mat(img, Rect(0,partHeight,img.cols, partHeight));
    bottomPart =    Mat(img, Rect(0,2 * partHeight,img.cols, partHeight));


}

