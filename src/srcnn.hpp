#ifndef SRCNN_HPP
#define SRCNN_HPP

#include <string>
#include <tuple>

#include "opencv2/opencv.hpp"

typedef std::tuple<int, int, int> Dim; // CHW (# channel, # height, # width)

typedef enum
{
    RELU
} ACTIVATION;

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

    void convolution(double *input, double *output, Dim inputDim,
        Dim outputDim, double *kernels, Dim kernelDim, int stride,
        double *bias, Dim biasDim);
    void activation(double *input, double *output, Dim inputDim, 
        ACTIVATION activationType);

    inline double relu_activate(double x) 
    {
        return x * (x > 0);
    }
};

#endif
