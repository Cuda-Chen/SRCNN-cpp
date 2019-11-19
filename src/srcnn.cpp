#include <string>

#include "opencv2/opencv.hpp"
#include "srcnn.hpp"

SRCNN::SRCNN()
{
}

void SRCNN::generate(std::string filename)
{
    using namespace cv;

    Mat img, gray;
    img = imread(filename, IMREAD_COLOR);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat bicubic;
    resize(gray, bicubic, Size(), this->scale, this->scale, INTER_CUBIC);

    namedWindow("input");
    imshow("input", img);
    waitKey(0);
    namedWindow("bicubic");
    imshow("bicubic", bicubic);
    waitKey(0);
}
