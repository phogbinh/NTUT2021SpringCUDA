#include "option.h"
#include <opencv2/opencv.hpp>

using namespace cv;

#ifndef PARALLEL
#define HEIGHT 512 /*原始圖片高*/
#define WIDTH 512 /*原始圖片寬*/
#define MAX_COLOR 255
#define GRADIENT_DIVISOR 2

/* Original Image RGB */
uchar Blue[HEIGHT][WIDTH];
uchar Green[HEIGHT][WIDTH];
uchar Red[HEIGHT][WIDTH];


/* new Image RGB after shift */
uchar nb[HEIGHT][WIDTH];
uchar ng[HEIGHT][WIDTH];
uchar nr[HEIGHT][WIDTH];


void Brite(uchar nr[HEIGHT][WIDTH], uchar ng[HEIGHT][WIDTH], uchar nb[HEIGHT][WIDTH],/* Image rgb after shift */
    uchar r[HEIGHT][WIDTH], uchar g[HEIGHT][WIDTH], uchar b[HEIGHT][WIDTH],/* Original Image rgb */
    int height, int width, int bri) /* Original height and width */
{
#ifdef ORIGINAL_CODE
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x >= width / 2) {
                if (Blue[y][x] + bri > 255) {
                    nb[y][x] = 255;
                }
                else {
                    nb[y][x] = Blue[y][x] + bri;
                }

                if (Green[y][x] + bri > 255) {
                    ng[y][x] = 255;
                }
                else {
                    ng[y][x] = Green[y][x] + bri;
                }

                if (Red[y][x] + bri > 255) {
                    nr[y][x] = 255;
                }
                else {
                    nr[y][x] = Red[y][x] + bri;
                }
            }
            else {
                nb[y][x] = Blue[y][x];
                ng[y][x] = Green[y][x];
                nr[y][x] = Red[y][x];
            }
        }
    }
#else
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            nr[i][j] = min(Red[i][j] + j / GRADIENT_DIVISOR, MAX_COLOR);
            nb[i][j] = min(Blue[i][j] + j / GRADIENT_DIVISOR, MAX_COLOR);
            ng[i][j] = min(Green[i][j] + j / GRADIENT_DIVISOR, MAX_COLOR);
        }
    }
#endif
}


int main()
{

    Mat img1 = imread("objects.jpg");


    /* Load Image RGB Values */
    for (int i = 0; i < img1.size().height; i++)
    {
        for (int j = 0; j < img1.size().width; j++)
        {
            Blue[i][j] = img1.at<Vec3b>(i, j)[0];
            Green[i][j] = img1.at<Vec3b>(i, j)[1];
            Red[i][j] = img1.at<Vec3b>(i, j)[2];
        }
    }

    ///* Image Brite */
    Brite(nr, ng, nb, /* Image rgb after shift */
        Red, Green, Blue, /* Original Image rgb */
        HEIGHT, WIDTH, 100 /* Original height and width */);


    /* Load shift Image RGB */
    for (int i = 0; i < img1.size().height; i++)
    {
        for (int j = 0; j < img1.size().width; j++)
        {
            img1.at<Vec3b>(i, j)[0] = nb[i][j];
            img1.at<Vec3b>(i, j)[1] = ng[i][j];
            img1.at<Vec3b>(i, j)[2] = nr[i][j];
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