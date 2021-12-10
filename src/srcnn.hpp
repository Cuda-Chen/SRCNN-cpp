#ifndef SRCNN_HPP
#define SRCNN_HPP

#include <string>
#include <vector>
#include <tuple>

#include "opencv2/opencv.hpp"
#include "datatype.hpp"

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

    void naiveConvolution(data_t *input, data_t *output, ImageDim inputDim,
        ImageDim outputDim, data_t *kernels, KernelDim kernelDim, int stride = 1,
        data_t *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0));
    void convolution(data_t *input, data_t *output, ImageDim inputDim,
        ImageDim outputDim, data_t *kernels, KernelDim kernelDim, int stride = 1,
        data_t *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0), data_t *workspace = NULL);
    void im2col(data_t *data_im, ImageDim imageDim, KernelDim kernelDim,
                int stride, int pad, data_t *data_col);
    void col2im(data_t *data_col, ImageDim imageDim, KernelDim kernelDim,
                int stride, int pad, data_t *data_im);
    data_t im2colGetPixel(data_t *im, ImageDim imageDim, 
                          int row, int col, int channel, int pad);
    void col2imAddPixel(data_t *im, ImageDim imageDim,
                        int row, int col, int channel, int pad, data_t value); 
    void matMul(data_t *out, data_t *kernel, data_t *in, data_t *bias,
                int kernel_row, int kernel_col, int in_row, int in_col); 
    void naiveGEMM(data_t *out, data_t *kernel, data_t *in,
                   int kernel_row, int kernel_col, int in_row, int in_col);
    void naiveGEMM_addBias(data_t *out, data_t *kernel, data_t *in, data_t *bias,
                           int kernel_row, int kernel_col, int in_row, int in_col);
    void tiledNVectorizedGEMM_addBias(data_t * __restrict__ pout, data_t * __restrict__ pkernel, data_t * __restrict__ pin, data_t *bias,
                           int kernel_row, int kernel_col, int in_row, int in_col);
    #ifdef ISX86
    void intrinsicGEMM_addBias(float *out, float *kernel, float *in, float *bias,
                           int kernel_row, int kernel_col, int in_row, int in_col);
    void gemm_nn(int M, int N, int K, float ALPHA,
                 float *A, int lda,
                 float *B, int ldb,
                 float *C, int ldc);
    void intrinsicGEMM_microkernel_addBias(float *out, float *kernel, float *in, float *bias,
                           int kernel_row, int kernel_col, int in_row, int in_col);
    #endif
    void transpose(data_t *out, data_t *in, int in_row, int in_col);

    void activation(data_t *input, data_t *output, ImageDim inputDim, 
        ACTIVATION activationType);

    inline data_t relu_activate(data_t x) 
    {
        return x * (x > 0);
    }

    void readConvWeights(std::string filename, data_t *weights, bool special = false, bool isReverse = false);
    void readConvWeights(std::string filename, data_t *weights, KernelDim kernelDim, WeightFormat format, bool special = false);
    void readBiasWeights(std::string filename, data_t *weights);

    // unit test functions
    void testConvolution(data_t *input, data_t *output, ImageDim inputDim,
        ImageDim outputDim, data_t *kernels, KernelDim kernelDim, int stride = 1,
        data_t *bias = NULL, ImageDim biasDim = std::make_tuple(0, 0, 0), 
        std::string outputConvWeightPath = NULL, std::string outputBiasWeightPath = NULL);
    void testReadConvWeights(std::string filename, std::string outputfile, data_t *weights, bool special = false, bool isReverse = false);
    void testReadBiasWeights(std::string filename, std::string outputfile, data_t *weights);    
    void testWriteWeights(std::string outputfile, data_t *weights, ImageDim imageDim);
    void testWriteWeights(std::string outputfile, data_t *weights, KernelDim kernelDim);
};

#endif
