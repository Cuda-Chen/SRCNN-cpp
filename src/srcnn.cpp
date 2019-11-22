#include <string>

#include "opencv2/opencv.hpp"
#include "srcnn.hpp"

using namespace std;
using namespace cv;

SRCNN::SRCNN()
{
}

void SRCNN::generate(string filename)
{
    this->img = imread(filename, IMREAD_COLOR);
    cvtColor(this->img, this->gray, COLOR_BGR2GRAY);
    resize(this->gray, this->bicubic, Size(), this->scale, this->scale, INTER_CUBIC);
}

void SRCNN::showOutput()
{
    namedWindow("input");
    imshow("input", this->img);
    waitKey(0);
    namedWindow("bicubic");
    imshow("bicubic", this->bicubic);
    waitKey(0);
}
