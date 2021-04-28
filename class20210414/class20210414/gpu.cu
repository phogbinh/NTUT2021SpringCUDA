#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#ifdef PARALLEL
__global__ void ProcessDataKernel(const uchar* const d_pFlowerData,
                                  const uchar* const d_pCarData,
                                  const int HEIGHT,
                                  const int WIDTH,
                                  const int CHANNELS_N,
                                  uchar* const d_pData)
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
            const int nIndex = (i * WIDTH + j) * CHANNELS_N + k;
            d_pData[nIndex] = (j < WIDTH / 2) ? d_pFlowerData[nIndex] : d_pCarData[nIndex];
        }
    }
}

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
    // CUDA prepare [d]evice data
    uchar* d_pFlowerData;
    uchar* d_pCarData;
    uchar* d_pData;

    cudaMalloc((void**)&d_pFlowerData, DATA_SIZE * sizeof(uchar));
    cudaMalloc((void**)&d_pCarData,    DATA_SIZE * sizeof(uchar));
    cudaMalloc((void**)&d_pData,       DATA_SIZE * sizeof(uchar));

    cudaMemcpy(d_pFlowerData, pFlowerData, DATA_SIZE * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pCarData,    pCarData,    DATA_SIZE * sizeof(uchar), cudaMemcpyHostToDevice);

    // CUDA process data
    ProcessDataKernel<<<16, 64>>>(d_pFlowerData, d_pCarData, HEIGHT, WIDTH, CHANNELS_N, d_pData);

    // CUDA write result onto flower image
    cudaMemcpy(pFlowerData, d_pData, DATA_SIZE * sizeof(uchar), cudaMemcpyDeviceToHost);

    // CUDA free [d]evice data
    cudaFree(d_pFlowerData);
    cudaFree(d_pCarData);
    cudaFree(d_pData);
    std::chrono::steady_clock::time_point kEnd = std::chrono::steady_clock::now();
    printf("Process data took me %ld nanoseconds.\n", std::chrono::duration_cast<std::chrono::nanoseconds>(kEnd - kBegin).count());

    // show image
    cv::imshow("Class 20210414", kFlower);
    cv::waitKey(0);

    return 0;
}
#endif
