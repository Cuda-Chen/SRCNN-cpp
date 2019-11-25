#ifndef SRCNN_HPP
#define SRCNN_HPP

#include <string>
#include <vector>
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
    SRCNN(std::string weights);
    void generate(std::string filename);
    void showOutput();
    void checkWeightStatus();
private:
    int scale = 2;
    cv::Mat img;
    cv::Mat gray;
    cv::Mat bicubic;
    cv::Mat output;

    std::string basePath = "model/";
    std::string weightsConv1 = "weights_conv1.txt";
    std::string weightsConv2 = "weights_conv2.txt";
    std::string weightsConv3 = "weights_conv3.txt";
    std::string biasConv1 = "biases_conv1.txt";
    std::string biasConv2 = "biases_conv2.txt";
    std::string biasConv3 = "biases_conv3.txt";
    std::vector<std::string> weights = {weightsConv1, weightsConv2,
        weightsConv3, biasConv1, biasConv2, biasConv3};

    void convolution(double *input, double *output, Dim inputDim,
        Dim outputDim, double *kernels, Dim kernelDim, int stride = 1,
        double *bias = NULL, Dim biasDim = std::make_tuple(0, 0, 0));
    void activation(double *input, double *output, Dim inputDim, 
        ACTIVATION activationType);

    inline double relu_activate(double x) 
    {
        return x * (x > 0);
    }
};

#endif
