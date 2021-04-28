#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef PARALLEL
#define HEIGHT 1024
#define WIDTH 1024
#define CHANNELS_N 3
#define BLOCKS_N 16
#define BLOCK_THREADS_N 64

__global__ void GetHorizontallyFlippedFrameKernel(uchar* const d_pOriginalFrame, uchar* const d_pFlippedFrame)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HEIGHT) // overflow
    {
        return;
    }
    for (int j = 0; j < WIDTH; ++j)
    {
        for (int k = 0; k < CHANNELS_N; ++k)
        {
            d_pFlippedFrame[(i * WIDTH + j) * CHANNELS_N + k] = d_pOriginalFrame[(i * WIDTH + (WIDTH - 1 - j)) * CHANNELS_N + k];
        }
    }
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

    // get original frame
    const unsigned int FRAME_SIZE = kCar.rows * kCar.step;
    uchar* const pFrame = (uchar*)malloc(FRAME_SIZE * sizeof(uchar));
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            for (int k = 0; k < CHANNELS_N; ++k)
            {
                pFrame[(i * WIDTH + j) * CHANNELS_N + k] = kCar.at<cv::Vec3b>(i, j)[k];
            }
        }
    }

    // CUDA prepare [d]evice frames
    uchar* d_pOriginalFrame;
    uchar* d_pFlippedFrame;

    cudaMalloc((void**)&d_pOriginalFrame, FRAME_SIZE * sizeof(uchar));
    cudaMalloc((void**)&d_pFlippedFrame,  FRAME_SIZE * sizeof(uchar));

    cudaMemcpy(d_pOriginalFrame, pFrame, FRAME_SIZE * sizeof(uchar), cudaMemcpyHostToDevice);

    // CUDA get [d]evice flipped frame
    GetHorizontallyFlippedFrameKernel<<<BLOCKS_N, BLOCK_THREADS_N>>>(d_pOriginalFrame, d_pFlippedFrame);

    // CUDA write result onto frame
    cudaMemcpy(pFrame, d_pFlippedFrame, FRAME_SIZE * sizeof(uchar), cudaMemcpyDeviceToHost);

    // CUDA free [d]evice frames
    cudaFree(d_pOriginalFrame);
    cudaFree(d_pFlippedFrame);

    // load frame to image
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            for (int k = 0; k < CHANNELS_N; ++k)
            {
                kCar.at<cv::Vec3b>(i, j)[k] = pFrame[(i * WIDTH + j) * CHANNELS_N + k];
            }
        }
    }

    // free frame
    free(pFrame);

    // stop time recorder
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