#include "option.h"
#include <opencv2/opencv.hpp>
#include <chrono>

#ifndef PARALLEL
int main()
{
    // read image
    cv::Mat kFlower = cv::imread("flower.jpg");
    cv::Mat kCar = cv::imread("car.jpg");

    // get image data
    const int HEIGHT = kFlower.rows;
    const int WIDTH = kFlower.cols;
    const int CHANNELS_N = kFlower.channels();
    const int DATA_SIZE = HEIGHT * WIDTH * kFlower.elemSize();
    uchar* const pFlowerData = (uchar*)kFlower.data;
    const uchar* const pCarData = (uchar*)kCar.data;

    std::chrono::steady_clock::time_point kBegin = std::chrono::steady_clock::now();
    // process data
    uchar* const pData = (uchar*)malloc(DATA_SIZE * sizeof(uchar));
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            for (int k = 0; k < CHANNELS_N; ++k)
            {
                const int nIndex = (i * WIDTH + j) * CHANNELS_N + k;
                pData[nIndex] = (j < WIDTH / 2) ? pFlowerData[nIndex] : pCarData[nIndex];
            }
        }
    }

    // write result onto flower image
    memcpy(pFlowerData, pData, DATA_SIZE * sizeof(uchar));

    // free data
    free(pData);
    std::chrono::steady_clock::time_point kEnd = std::chrono::steady_clock::now();
    printf("Process data took me %ld nanoseconds.\n", std::chrono::duration_cast<std::chrono::nanoseconds>(kEnd - kBegin).count());

    // show image
    cv::imshow("Class 20210414", kFlower);
    cv::waitKey(0);

    return 0;
}
#endif
