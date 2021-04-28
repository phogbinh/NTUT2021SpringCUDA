#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef PARALLEL
#define HEIGHT 1024
#define WIDTH 1024
#define BLOCKS_N 16
#define BLOCK_THREADS_N 64

uchar** const GetImageColorPointer()
{
    uchar** const pColor = (uchar**)malloc(HEIGHT * sizeof(uchar*));
    for (int i = 0; i < HEIGHT; ++i)
    {
        pColor[i] = (uchar*)malloc(WIDTH * sizeof(uchar));
    }
    return pColor;
}

void FreeImageColorPointer(uchar** const pColor)
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        free(pColor[i]);
    }
    free(pColor);
}

uchar** const GetImageOriginalColorPointerCuda(uchar** const pColor)
{
    uchar** d_pColor;
    cudaMalloc((void***)&d_pColor, HEIGHT * sizeof(uchar*));
    for (int i = 0; i < HEIGHT; ++i)
    {
        cudaMalloc((void**)&d_pColor[i], WIDTH * sizeof(uchar)); /* BUG HERE */
        cudaMemcpy(d_pColor[i], pColor[i], WIDTH * sizeof(uchar), cudaMemcpyHostToDevice);
    }
    return d_pColor;
}

uchar** const GetImageColorPointerCuda()
{
    uchar** d_pColor;
    cudaMalloc((void***)&d_pColor, HEIGHT * sizeof(uchar*));
    for (int i = 0; i < HEIGHT; ++i)
    {
        cudaMalloc((void**)&d_pColor[i], WIDTH * sizeof(uchar));
    }
    return d_pColor;
}

void FreeImageColorPointerCuda(uchar** const d_pColor)
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        cudaFree(d_pColor[i]);
    }
    cudaFree(d_pColor);
}

__global__ void GetHorizontallyFlippedRgbKernel(uchar** const d_pOriginalRed,
                                                uchar** const d_pOriginalGreen,
                                                uchar** const d_pOriginalBlue,
                                                uchar** const d_pFlippedRed,
                                                uchar** const d_pFlippedGreen,
                                                uchar** const d_pFlippedBlue)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HEIGHT) // overflow
    {
        return;
    }
    for (int j = 0; j < WIDTH; ++j)
    {
        int x = i;
        int y = WIDTH - 1 - j;
        d_pFlippedRed[i][j] = d_pOriginalRed[x][y];
        d_pFlippedGreen[i][j] = d_pOriginalGreen[x][y];
        d_pFlippedBlue[i][j] = d_pOriginalBlue[x][y];
    }
}

int main()
{
    // read image
    cv::Mat kCar = cv::imread("car.jpg");

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

    // CUDA get [d]evice original rgb
    uchar** const d_pOriginalRed = GetImageOriginalColorPointerCuda(pOriginalRed);
    uchar** const d_pOriginalGreen = GetImageOriginalColorPointerCuda(pOriginalGreen);
    uchar** const d_pOriginalBlue = GetImageOriginalColorPointerCuda(pOriginalBlue);

    // CUDA prepare [d]evice flipped rgb
    uchar** const d_pFlippedRed = GetImageColorPointerCuda();
    uchar** const d_pFlippedGreen = GetImageColorPointerCuda();
    uchar** const d_pFlippedBlue = GetImageColorPointerCuda();

    // CUDA get [d]evice flipped rgb
    GetHorizontallyFlippedRgbKernel<<<BLOCKS_N, BLOCK_THREADS_N>>>(d_pOriginalRed,
                                                                   d_pOriginalGreen,
                                                                   d_pOriginalBlue,
                                                                   d_pFlippedRed,
                                                                   d_pFlippedGreen,
                                                                   d_pFlippedBlue);

    // CUDA free [d]evice original rgb
    FreeImageColorPointerCuda(d_pOriginalRed);
    FreeImageColorPointerCuda(d_pOriginalGreen);
    FreeImageColorPointerCuda(d_pOriginalBlue);

    // CUDA load [d]evice flipped rgb to original rgb
    for (int i = 0; i < HEIGHT; ++i)
    {
        cudaMemcpy(pOriginalRed, d_pFlippedRed, WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost);
        cudaMemcpy(pOriginalGreen, d_pFlippedGreen, WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost);
        cudaMemcpy(pOriginalBlue, d_pFlippedBlue, WIDTH * sizeof(uchar), cudaMemcpyDeviceToHost);
    }

    // CUDA free [d]evice flipped rgb
    FreeImageColorPointerCuda(d_pFlippedRed);
    FreeImageColorPointerCuda(d_pFlippedGreen);
    FreeImageColorPointerCuda(d_pFlippedBlue);

    // load original rgb to image
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            kCar.at<cv::Vec3b>(i, j)[0] = pOriginalBlue[i][j];
            kCar.at<cv::Vec3b>(i, j)[1] = pOriginalGreen[i][j];
            kCar.at<cv::Vec3b>(i, j)[2] = pOriginalRed[i][j];
        }
    }

    // free original rgb
    FreeImageColorPointer(pOriginalRed);
    FreeImageColorPointer(pOriginalGreen);
    FreeImageColorPointer(pOriginalBlue);

    // show image
    cv::imshow("Image Horizontal Flip", kCar);
    cv::waitKey(0);

    return 0;
}
#endif