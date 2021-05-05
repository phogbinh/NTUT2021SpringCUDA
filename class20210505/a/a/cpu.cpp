#include "option.h"
#include <opencv2/opencv.hpp>

#ifndef PARALLEL
using namespace cv;

#define HEIGHT 512 /*原始圖片高*/
#define WIDTH 512 /*原始圖片寬*/

#define OFFSET 30  /*偏移量_高*/
#define offset_x 30  /*偏移量_寬*/


/* Original Image RGB */
uchar Blue[HEIGHT][WIDTH];
uchar Green[HEIGHT][WIDTH];
uchar Red[HEIGHT][WIDTH];


/* new Image RGB after shift */
uchar nb[HEIGHT][WIDTH];
uchar ng[HEIGHT][WIDTH];
uchar nr[HEIGHT][WIDTH];


void bmp_shift(uchar nr[HEIGHT][WIDTH], uchar ng[HEIGHT][WIDTH], uchar nb[HEIGHT][WIDTH],/* Image rgb after shift */
    uchar kOriginalRed[HEIGHT][WIDTH], uchar kOriginalGreen[HEIGHT][WIDTH], uchar kOriginalBlue[HEIGHT][WIDTH],/* Original Image rgb */
    int height, int width, int oy, int ox) /* Original height and width */
{
    const unsigned int SEGMENT_WIDTH = WIDTH / 5;
    for (int i = 0; i < HEIGHT; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            const unsigned int SEGMENT_ID = j / SEGMENT_WIDTH;
            if (SEGMENT_ID == 1 || SEGMENT_ID == 3)
            {
                const int x = i - OFFSET;
                if (x >= 0)
                {
                    nr[i][j] = kOriginalRed[x][j];
                    ng[i][j] = kOriginalGreen[x][j];
                    nb[i][j] = kOriginalBlue[x][j];
                }
            }
            else
            {
                const unsigned int x = i + OFFSET;
                if (x < HEIGHT)
                {
                    nr[i][j] = kOriginalRed[x][j];
                    ng[i][j] = kOriginalGreen[x][j];
                    nb[i][j] = kOriginalBlue[x][j];
                }
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

    ///* Image shift */
    bmp_shift(nr, ng, nb, /* Image rgb after shift */
        Red, Green, Blue, /* Original Image rgb */
        HEIGHT, WIDTH, OFFSET, offset_x /* Original height and width */);


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

    // show the image
    imshow("mainWin", img1);

    // wait for a key
    waitKey(0);

    return 0;
}
#endif