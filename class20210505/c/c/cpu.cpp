#include "option.h"
#include <opencv2/opencv.hpp>

#ifndef PARALLEL
using namespace cv;

#define HEIGHT 512
#define WIDTH 512
#define CENTER_X (HEIGHT / 2)
#define CENTER_Y (WIDTH / 2)
#define RADIUS 50
#define RADIUS_SQUARED (RADIUS * RADIUS)

/* Original Image RGB */
uchar Blue[HEIGHT][WIDTH];
uchar Green[HEIGHT][WIDTH];
uchar Red[HEIGHT][WIDTH];


/* new Image RGB after shift */
uchar nb[HEIGHT][WIDTH];
uchar ng[HEIGHT][WIDTH];
uchar nr[HEIGHT][WIDTH];


void negative(uchar nr[HEIGHT][WIDTH], uchar ng[HEIGHT][WIDTH], uchar nb[HEIGHT][WIDTH],/* Image rgb after shift */
    uchar r[HEIGHT][WIDTH], uchar g[HEIGHT][WIDTH], uchar b[HEIGHT][WIDTH],/* Original Image rgb */
    int height, int width) /* Original height and width */
{
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            const unsigned int distanceSquared = (i - CENTER_X) * (i - CENTER_X) + (j - CENTER_Y) * (j - CENTER_Y);
            const unsigned int encapsulatingCircleRadiusMultiplier = ceil(sqrt(distanceSquared * 1.0 / RADIUS_SQUARED));
            const unsigned int innerRadiusSquared = (encapsulatingCircleRadiusMultiplier - 1) * (encapsulatingCircleRadiusMultiplier - 1) * RADIUS_SQUARED;
            if (encapsulatingCircleRadiusMultiplier % 2 == 1 && innerRadiusSquared <= distanceSquared)
            {
                nb[i][j] = 255 - Blue[i][j];
                ng[i][j] = 255 - Green[i][j];
                nr[i][j] = 255 - Red[i][j];
            }
            else
            {
                nb[i][j] = Blue[i][j];
                ng[i][j] = Green[i][j];
                nr[i][j] = Red[i][j];
            }
        }
    }
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
    negative(nr, ng, nb, /* Image rgb after shift */
        Red, Green, Blue, /* Original Image rgb */
        HEIGHT, WIDTH /* Original height and width */);


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