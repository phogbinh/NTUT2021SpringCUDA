#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

#ifndef PARALLEL
#define HEIGHT 1024
#define WIDTH 1024
#define STRIP_WIDTH 30
#define STRIP_R 255
#define STRIP_G 0
#define STRIP_B 0

uchar** const GetImageColorPointer()
{
    uchar** const pColor = (uchar**)malloc(HEIGHT * sizeof(uchar*));
    for (int i = 0; i < HEIGHT; ++i)
    {
        pColor[i] = (uchar*)malloc(WIDTH * sizeof(uchar));
    }
    return pColor;
}

void GetDiagonallyStrippedRgb(uchar** const pOriginalRed,
                              uchar** const pOriginalGreen,
                              uchar** const pOriginalBlue,
                              uchar** const pStrippedRed,
                              uchar** const pStrippedGreen,
                              uchar** const pStrippedBlue)
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            if (i % STRIP_WIDTH == j % STRIP_WIDTH)
            {
                pStrippedRed[i][j] = STRIP_R;
                pStrippedGreen[i][j] = STRIP_G;
                pStrippedBlue[i][j] = STRIP_B;
            }
            else
            {
                pStrippedRed[i][j] = pOriginalRed[i][j];
                pStrippedGreen[i][j] = pOriginalGreen[i][j];
                pStrippedBlue[i][j] = pOriginalBlue[i][j];
            }
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

    // get stripped rgb
    uchar** const pStrippedRed = GetImageColorPointer();
    uchar** const pStrippedGreen = GetImageColorPointer();
    uchar** const pStrippedBlue = GetImageColorPointer();
    GetDiagonallyStrippedRgb(pOriginalRed,
                             pOriginalGreen,
                             pOriginalBlue,
                             pStrippedRed,
                             pStrippedGreen,
                             pStrippedBlue);

    // free original rgb
    FreeImageColorPointer(pOriginalRed);
    FreeImageColorPointer(pOriginalGreen);
    FreeImageColorPointer(pOriginalBlue);

    // load stripped rgb to image
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            kCar.at<cv::Vec3b>(i, j)[0] = pStrippedBlue[i][j];
            kCar.at<cv::Vec3b>(i, j)[1] = pStrippedGreen[i][j];
            kCar.at<cv::Vec3b>(i, j)[2] = pStrippedRed[i][j];
        }
    }

    // free stripped rgb
    FreeImageColorPointer(pStrippedRed);
    FreeImageColorPointer(pStrippedGreen);
    FreeImageColorPointer(pStrippedBlue);

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
    cv::imshow("Image Diagonal Strip", kCar);
    cv::waitKey(0);

    return 0;
}
#endif