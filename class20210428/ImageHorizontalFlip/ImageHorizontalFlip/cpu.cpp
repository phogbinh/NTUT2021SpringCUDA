#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

#ifndef PARALLEL
#define HEIGHT 1024
#define WIDTH 1024

uchar** const GetImageColorPointer()
{
    uchar** const pColor = (uchar**)malloc(HEIGHT * sizeof(uchar*));
    for (int i = 0; i < HEIGHT; ++i)
    {
        pColor[i] = (uchar*)malloc(WIDTH * sizeof(uchar));
    }
    return pColor;
}

void GetHorizontallyFlippedRgb(uchar** const pOriginalRed,
                               uchar** const pOriginalGreen,
                               uchar** const pOriginalBlue,
                               uchar** const pFlippedRed,
                               uchar** const pFlippedGreen,
                               uchar** const pFlippedBlue)
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            int x = i;
            int y = WIDTH - 1 - j;
            pFlippedRed[i][j] = pOriginalRed[x][y];
            pFlippedGreen[i][j] = pOriginalGreen[x][y];
            pFlippedBlue[i][j] = pOriginalBlue[x][y];
        }
    }
}

void FreeImageColorPointer(uchar** const pColor)
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        free(pColor[i]);
    }
    free(pColor);
}

int main()
{
    // read image
    cv::Mat kCar = cv::imread("car.jpg");

    // start time recorder
    cudaEvent_t kStart;
    cudaEvent_t kStop;
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);
    cudaEventRecord(kStart, 0);
    float* d_pTimeDump;
    cudaMalloc((void**)&d_pTimeDump, sizeof(float));

    // get original rgb
    uchar** const pOriginalRed = GetImageColorPointer();
    uchar** const pOriginalGreen = GetImageColorPointer();
    uchar** const pOriginalBlue = GetImageColorPointer();
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            pOriginalBlue[i][j] = kCar.at<cv::Vec3b>(i, j)[0];
            pOriginalGreen[i][j] = kCar.at<cv::Vec3b>(i, j)[1];
            pOriginalRed[i][j] = kCar.at<cv::Vec3b>(i, j)[2];
        }
    }

    // get flipped rgb
    uchar** const pFlippedRed = GetImageColorPointer();
    uchar** const pFlippedGreen = GetImageColorPointer();
    uchar** const pFlippedBlue = GetImageColorPointer();
    GetHorizontallyFlippedRgb(pOriginalRed,
                              pOriginalGreen,
                              pOriginalBlue,
                              pFlippedRed,
                              pFlippedGreen,
                              pFlippedBlue);

    // free original rgb
    FreeImageColorPointer(pOriginalRed);
    FreeImageColorPointer(pOriginalGreen);
    FreeImageColorPointer(pOriginalBlue);

    // load flipped rgb to image
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            kCar.at<cv::Vec3b>(i, j)[0] = pFlippedBlue[i][j];
            kCar.at<cv::Vec3b>(i, j)[1] = pFlippedGreen[i][j];
            kCar.at<cv::Vec3b>(i, j)[2] = pFlippedRed[i][j];
        }
    }

    // free flipped rgb
    FreeImageColorPointer(pFlippedRed);
    FreeImageColorPointer(pFlippedGreen);
    FreeImageColorPointer(pFlippedBlue);

    // stop time recorder
    cudaFree(d_pTimeDump);
    cudaEventRecord(kStop, 0);
    cudaEventSynchronize(kStop);
    float fTimeMs = 0.f;
    cudaEventElapsedTime(&fTimeMs, kStart, kStop);
    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);
    printf("Process data took me %f milliseconds.\n", fTimeMs);

    // show image
    cv::imshow("Image Horizontal Flip", kCar);
    cv::waitKey(0);

    return 0;
}
#endif