#ifndef SRCNN_HPP
#define SRCNN_HPP

#include <string>
#include <vector>
#include <tuple>

#include "opencv2/opencv.hpp"

typedef std::tuple<int, int, int> ImageDim; // CHW (# channel, # height, # width)
typedef std::tuple<int, int, int, int> KernelDim; // NCHW (# outputChannel, # inputChannel, # height, # width)

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
    int getTotalDimension(ImageDim dim);
    int getTotalDimension(KernelDim dim);

    // Unit test functions, do not use them.
    void checkWeightStatus();
    void testImageConv(std::string filename);
    void testConv();
private:
    int scale = 2;
    cv::Mat img;
    cv::Mat gray;
    cv::Mat downsample;
    cv::Mat bicubic;
    cv::Mat output;

    std::string basePath = "./model/";
    //std::string basePath = "./";
    std::string weightsConv1 = "weights_conv1.txt";
    std::string weightsConv2 = "weights_conv2.txt";
    std::string weightsConv3 = "weights_conv3.txt";
    std::string biasConv1 = "biases_conv1.txt";
    std::string biasConv2 = "biases_conv2.txt";
    std::string biasConv3 = "biases_conv3.txt";
    std::vector<std::string> weights = {weightsConv1, weightsConv2,
        weightsConv3, biasConv1, biasConv2, biasConv3};

    void convolution(double *input, double *output, ImageDim inputDim,
        ImageDim outputDim, double *kernels, KernelDim kernelDim, int stride = 1,
        double *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0));
    void im2col(double *data_im, ImageDim imageDim, KernelDim kernelDim,
                int stride, int pad, double *data_col);
    void col2im(double *data_col, ImageDim imageDim, KernelDim kernelDim,
                int stride, int pad, double *data_im);
    double im2colGetPixel(double *im, ImageDim imageDim, 
                          int row, int col, int channel, int pad);
    void col2imAddPixel(double *im, ImageDim imageDim,
                        int row, int col, int channel, int pad, double value);  
    void naiveGEMM(double *data_col, double *kernel_col, int col_size);
    void addBias(double *data_col, double *bias_col, int col_size); 
    void activation(double *input, double *output, ImageDim inputDim, 
        ACTIVATION activationType);

    inline double relu_activate(double x) 
    {
        return x * (x > 0);
    }

    void readConvWeights(std::string filename, double *weights, bool special = false, bool isReverse = false);
    void readBiasWeights(std::string filename, double *weights);

    // unit test functions
    void testConvolution(double *input, double *output, ImageDim inputDim,
        ImageDim outputDim, double *kernels, KernelDim kernelDim, int stride = 1,
        double *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0), 
        std::string outputConvWeightPath = NULL, std::string outputBiasWeightPath = NULL);
    void testReadConvWeights(std::string filename, std::string outputfile, double *weights, bool special = false, bool isReverse = false);
    void testReadBiasWeights(std::string filename, std::string outputfile, double *weights);
};

#endif
