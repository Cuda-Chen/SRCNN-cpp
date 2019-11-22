#ifndef SRCNN_HPP
#define SRCNN_HPP

#include <string>

#include "opencv2/opencv.hpp"

class SRCNN
{
public:
    SRCNN();
    void generate(std::string filename);
    void showOutput();
private:
    int scale = 2;
    cv::Mat img;
    cv::Mat gray;
    cv::Mat bicubic;
    cv::Mat output;


};

#endif
