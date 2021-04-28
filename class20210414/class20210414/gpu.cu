#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCKS_N 32
#define BLOCK_THREADS_N 64

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

    // start time recorder
    cudaEvent_t kStart;
    cudaEvent_t kStop;
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);
    cudaEventRecord(kStart, 0);

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
    ProcessDataKernel<<<BLOCKS_N, BLOCK_THREADS_N>>>(d_pFlowerData, d_pCarData, HEIGHT, WIDTH, CHANNELS_N, d_pData);

    // CUDA write result onto flower image
    cudaMemcpy(pFlowerData, d_pData, DATA_SIZE * sizeof(uchar), cudaMemcpyDeviceToHost);

    // CUDA free [d]evice data
    cudaFree(d_pFlowerData);
    cudaFree(d_pCarData);
    cudaFree(d_pData);

    // stop time recorder
    cudaEventRecord(kStop, 0);
    cudaEventSynchronize(kStop);
    float fTimeMs = 0.f;
    cudaEventElapsedTime(&fTimeMs, kStart, kStop);
    cudaEventDestroy(kStart);
    cudaEventDestroy(kStop);
    printf("Process data took me %f milliseconds.\n", fTimeMs);

    // show image
    cv::imshow("Class 20210414", kFlower);
    cv::waitKey(0);

    return 0;
}
#endif
