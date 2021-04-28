#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"

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

    // start time recorder
    cudaEvent_t kStart;
    cudaEvent_t kStop;
    cudaEventCreate(&kStart);
    cudaEventCreate(&kStop);
    cudaEventRecord(kStart, 0);

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