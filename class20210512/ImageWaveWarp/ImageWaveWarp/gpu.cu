#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;

#define HEIGHT 512 /*原始圖片高*/
#define WIDTH 512 /*原始圖片寬*/
#define LEFT 150
#define RIGHT 250
#define CHANNELS_N 3
#define PI 3.14
#define WAVE_PERIOD 20.0
#define WAVE_MAX 5.0

__global__ void swirl_Kernel(uchar* d_frame_out, uchar* d_frame_in,
    int height, int width, int centery, int centerx) /* Original height and width */
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= HEIGHT)
    {
        return;
    }
    for (int j = 0; j < WIDTH; ++j)
    {
        if (j < LEFT || j > RIGHT)
        {
            for (int k = 0; k < CHANNELS_N; ++k)
            {
                d_frame_out[(i * WIDTH + j) * CHANNELS_N + k] = d_frame_in[(i * WIDTH + j) * CHANNELS_N + k];
            }
            continue;
        }

        double dx = WAVE_MAX * sin(2 * PI / WAVE_PERIOD * i);
        int newX = i + dx;
        int newY = j;

        if (newX < 0)
        {
            newX = 0;
        }
        if (newX > HEIGHT - 1)
        {
            newX = HEIGHT - 1;
        }
        if (newY < 0)
        {
            newY = 0;
        }
        if (newY > WIDTH - 1)
        {
            newY = WIDTH - 1;
        }
        for (int k = 0; k < CHANNELS_N; ++k)
        {
            d_frame_out[(i * WIDTH + j) * CHANNELS_N + k] = d_frame_in[(newX * WIDTH + newY) * CHANNELS_N + k];
        }
    }
}


int main()
{

    Mat img1 = imread("lena.jpg");
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
    swirl_Kernel << <16, 64 >> > (d_frame_out, d_frame_in,
        HEIGHT, WIDTH, HEIGHT / 2, WIDTH / 2/* Original height and width */);

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
