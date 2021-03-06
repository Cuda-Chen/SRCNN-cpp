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
    void outputImage();
    int getTotalDimension(ImageDim dim);
    int getTotalDimension(KernelDim dim);

    // Unit test functions, do not use them.
    void checkWeightStatus();
    void testImageConv(std::string filename);
    void testConv1Channel();
    void testConv3Channels();
    void testTranspose();
    void testReadAndTranspose();
    void testReadWeightFormat();

private:
    int scale = 2;
    cv::Mat img;
    cv::Mat gray;
    cv::Mat downsample;
    cv::Mat bicubic;
    cv::Mat output;
    enum WeightFormat
    {
        NCHW,
        NHWC,
        CHWN,
        NCWH
    };

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

    void naiveConvolution(double *input, double *output, ImageDim inputDim,
        ImageDim outputDim, double *kernels, KernelDim kernelDim, int stride = 1,
        double *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0));
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
    void matMul(double *out, double *kernel, double *in, double *bias,
                int kernel_row, int kernel_col, int in_row, int in_col); 
    void naiveGEMM(double *out, double *kernel, double *in,
                   int kernel_row, int kernel_col, int in_row, int in_col);
    void naiveGEMM_addBias(double *out, double *kernel, double *in, double *bias,
                           int kernel_row, int kernel_col, int in_row, int in_col);
    void transpose(double *out, double *in, int in_row, int in_col);

    void activation(double *input, double *output, ImageDim inputDim, 
        ACTIVATION activationType);

    inline double relu_activate(double x) 
    {
        return x * (x > 0);
    }

    void readConvWeights(std::string filename, double *weights, bool special = false, bool isReverse = false);
    void readConvWeights(std::string filename, double *weights, KernelDim kernelDim, WeightFormat format, bool special = false);
    void readBiasWeights(std::string filename, double *weights);

    // unit test functions
    void testConvolution(double *input, double *output, ImageDim inputDim,
        ImageDim outputDim, double *kernels, KernelDim kernelDim, int stride = 1,
        double *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0), 
        std::string outputConvWeightPath = NULL, std::string outputBiasWeightPath = NULL);
    void testReadConvWeights(std::string filename, std::string outputfile, double *weights, bool special = false, bool isReverse = false);
    void testReadBiasWeights(std::string filename, std::string outputfile, double *weights);    
    void testWriteWeights(std::string outputfile, double *weights, ImageDim imageDim);
    void testWriteWeights(std::string outputfile, double *weights, KernelDim kernelDim);
};

#endif
