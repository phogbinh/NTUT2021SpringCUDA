#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef PARALLEL
using namespace cv;

#define HEIGHT 512 /*原始圖片高*/
#define WIDTH 512 /*原始圖片寬*/
#define CENTER_X (HEIGHT / 2)
#define CENTER_Y (WIDTH / 2)
#define RADIUS 50
#define RADIUS_SQUARED (RADIUS * RADIUS)

__global__ void negative_Kernel(uchar* d_frame_out, uchar* d_frame_in)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= HEIGHT)
    {
        return;
    }
    for (int j = 0; j < WIDTH; ++j)
    {
        const unsigned int distanceSquared = (i - CENTER_X) * (i - CENTER_X) + (j - CENTER_Y) * (j - CENTER_Y);
        const unsigned int encapsulatingCircleRadiusMultiplier = ceil(sqrt(distanceSquared * 1.0 / RADIUS_SQUARED));
        const unsigned int innerRadiusSquared = (encapsulatingCircleRadiusMultiplier - 1) * (encapsulatingCircleRadiusMultiplier - 1) * RADIUS_SQUARED;
        if (encapsulatingCircleRadiusMultiplier % 2 == 1 && innerRadiusSquared <= distanceSquared)
        {
            for (int k = 0; k < 3; ++k)
            {
                d_frame_out[(i * WIDTH + j) * 3 + k] = 255 - d_frame_in[(i * WIDTH + j) * 3 + k];
            }
        }
        else
        {
            for (int k = 0; k < 3; ++k)
            {
                d_frame_out[(i * WIDTH + j) * 3 + k] = d_frame_in[(i * WIDTH + j) * 3 + k];
            }
        }
    }
}


int main()
{

    Mat img1 = imread("objects.jpg");
    int img1_size = img1.rows * img1.step;
    uchar* frame1 = (uchar*)calloc(img1_size, sizeof(uchar));

    /* Load Image RGB Values */
    for (int i = 0; i < img1.size().height; i++)
    {
        for (int j = 0; j < img1.size().width; j++)
        {
            for (int z = 0; z < 3; z++) {
                frame1[(i * img1.size().width + j) * 3 + z] = img1.at<Vec3b>(i, j)[z];
            }
        }
    }

    uchar* d_frame_in;
    uchar* d_frame_out;

    cudaMalloc((void**)&d_frame_in, sizeof(uchar) * img1_size);
    cudaMalloc((void**)&d_frame_out, sizeof(uchar) * img1_size);

    cudaMemcpy(d_frame_in, frame1, sizeof(uchar) * img1_size, cudaMemcpyHostToDevice);

    ///* Image shift */
    negative_Kernel << <16, 64 >> > (d_frame_out, d_frame_in);

    cudaMemcpy(frame1, d_frame_out, sizeof(uchar) * img1_size, cudaMemcpyDeviceToHost);



    /* Load shift Image RGB */
    for (int i = 0; i < img1.size().height; i++)
    {
        for (int j = 0; j < img1.size().width; j++)
        {
            for (int z = 0; z < 3; z++) {
                img1.at<Vec3b>(i, j)[z] = frame1[(i * img1.size().width + j) * 3 + z];
            }
        }
    }

    // create a window
    namedWindow("mainWin", WINDOW_AUTOSIZE);
    moveWindow("mainWin", 100, 100);

    // show the image
    imshow("mainWin", img1);

    // wait for a key
    waitKey(0);

    return 0;
}
#endif