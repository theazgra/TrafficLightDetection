//
// Created by azgra on 10.2.18.
//

#include "OpenCvUtils.h"
/*
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

float get_brightness_value(cv::Mat1b & maskedImg)
{
    long totalPixelCount = maskedImg.rows * (maskedImg.cols / 3);
    long whitePixelCount = 0;

    dlib::mutex mutexLock;

    if (!maskedImg.isContinuous())
    {
        maskedImg = maskedImg.clone();
    }

    dlib::parallel_for(0, totalPixelCount, [&](long i){
        uchar pixel = maskedImg.data[i];

        //White pixel
        if (pixel == 255)
        {
            dlib::auto_mutex lock(mutexLock);
            ++whitePixelCount;
        }
    });

    float brightness = ((float)(whitePixelCount)) / ((float)(totalPixelCount));

    return brightness;
}
*/

int thread_count = 10;
dlib::mutex count_mutex;
dlib::signaler count_signaler(count_mutex);

void thread(void * param)
{
    Str * test = (Str*)param;
    std::cout << test->val << std::endl;

    count_signaler.signal();
}



TLState get_traffic_light_state()
{
    //using namespace cv;
    Str s;
    s.val = 5;

    dlib::create_new_thread(thread, (void*)&s);

    while (thread_count > 0)
        count_signaler.wait();

    //delete s;

    return Inactive;
/*
    //It is best probably to mask white, brightest spots
    Scalar lowerWhite = Scalar(0,0,225);
    Scalar upperWhite = Scalar(255,255,255);

    Scalar lower_red_0 = Scalar(0, 100, 100);
    Scalar upper_red_0 = Scalar(15, 255, 255);
    Scalar lower_red_1 = Scalar(180 - 15, 100, 100);
    Scalar upper_red_1 = Scalar(180, 255, 255);

    Scalar lowerGreen = Scalar(44, 71, 82);
    Scalar upperGreen = Scalar(105, 255, 255);

    Scalar lowerOrange = Scalar(0, 26, 40);
    Scalar upperOrange = Scalar(67, 255, 255);


    Mat topPart, middlePart, bottomPart;
    Mat1b topMask, middleMask, bottomMask;

    int partHeight = img.rows / 3;

    topPart =       Mat(img, Rect(0,0,img.cols, partHeight));
    middlePart =    Mat(img, Rect(0,partHeight,img.cols, partHeight));
    bottomPart =    Mat(img, Rect(0,2 * partHeight,img.cols, partHeight));

*/
}

