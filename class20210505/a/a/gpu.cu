#include "option.h"
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef PARALLEL
using namespace cv;

//#define ORIGINAL_CODE

#define HEIGHT 512 /*原始圖片高*/
#define WIDTH 512 /*原始圖片寬*/

#define OFFSET 30  /*偏移量_高*/
#define offset_x 30  /*偏移量_寬*/



__global__ void shift_Kernel(uchar* d_frame_out, uchar* d_frame_in,
    int height, int width, int oy, int ox) /* Original height and width */
{
#ifdef ORIGINAL_CODE
    int y, x;
    int nx = threadIdx.x + blockIdx.x * blockDim.x;

    if (nx > 0 && nx < width) {
        for (int ny = 0; ny < height; ny++)
        {
            x = nx - ox, y = ny - oy;
            if (x < width && y < height && x >= 0 && y >= 0) {
                d_frame_out[(ny * width + nx) * 3 + 0] = d_frame_in[(y * width + x) * 3 + 0];
                d_frame_out[(ny * width + nx) * 3 + 1] = d_frame_in[(y * width + x) * 3 + 1];
                d_frame_out[(ny * width + nx) * 3 + 2] = d_frame_in[(y * width + x) * 3 + 2];
            }
        }
    }
#else
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= HEIGHT)
    {
        return;
    }
    const unsigned int SEGMENT_WIDTH = WIDTH / 5;
    for (int j = 0; j < WIDTH; ++j)
    {
        const unsigned int SEGMENT_ID = j / SEGMENT_WIDTH;
        if (SEGMENT_ID == 1 || SEGMENT_ID == 3)
        {
            const int x = i - OFFSET;
            if (x >= 0)
            {
                d_frame_out[(i * WIDTH + j) * 3 + 0] = d_frame_in[(x * WIDTH + j) * 3 + 0];
                d_frame_out[(i * WIDTH + j) * 3 + 1] = d_frame_in[(x * WIDTH + j) * 3 + 1];
                d_frame_out[(i * WIDTH + j) * 3 + 2] = d_frame_in[(x * WIDTH + j) * 3 + 2];
            }
        }
        else
        {
            const unsigned int x = i + OFFSET;
            if (x < HEIGHT)
            {
                d_frame_out[(i * WIDTH + j) * 3 + 0] = d_frame_in[(x * WIDTH + j) * 3 + 0];
                d_frame_out[(i * WIDTH + j) * 3 + 1] = d_frame_in[(x * WIDTH + j) * 3 + 1];
                d_frame_out[(i * WIDTH + j) * 3 + 2] = d_frame_in[(x * WIDTH + j) * 3 + 2];
            }
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
    shift_Kernel << <16, 64 >> > (d_frame_out, d_frame_in,
        HEIGHT, WIDTH, OFFSET, offset_x /* Original height and width */);

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