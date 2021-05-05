#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#define HEIGHT 512 /*原始圖片高*/
#define WIDTH 512 /*原始圖片寬*/
#define MAX_COLOR 255
#define GRADIENT_DIVISOR 2

__global__ void Brite_Kernel(uchar* d_frame_out, uchar* d_frame_in,
    int height, int width, int bri) /* Original height and width */
{
#ifdef ORIGINAL_CODE
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x > 0 && x < width) {
        for (int y = 0; y < height; y++)
        {
            for (int z = 0; z < 3; z++) {
                if (x >= width / 2) {
                    if (d_frame_in[(y * width + x) * 3 + z] + bri > 255) {
                        d_frame_out[(y * width + x) * 3 + z] = 255;
                    }
                    else {
                        d_frame_out[(y * width + x) * 3 + z] = d_frame_in[(y * width + x) * 3 + z] + bri;
                    }
                }
                else {
                    d_frame_out[(y * width + x) * 3 + z] = d_frame_in[(y * width + x) * 3 + z];
                }
            }
        }
    }
#else
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= HEIGHT)
    {
        return;
    }
    for (int j = 0; j < WIDTH; ++j)
    {
        for (int k = 0; k < 3; ++k)
        {
            int nColor = d_frame_in[(i * width + j) * 3 + k] + j / GRADIENT_DIVISOR;
            d_frame_out[(i * width + j) * 3 + k] = (nColor < MAX_COLOR) ? nColor : MAX_COLOR;
        }
    }
#endif
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
    Brite_Kernel << <16, 64 >> > (d_frame_out, d_frame_in,
        HEIGHT, WIDTH, 100 /* Original height and width */);

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
