#include <iostream>
#include <iomanip>
#include <string>
#include <tuple>
#include <fstream>
#include <cassert>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "srcnn.hpp"
#include "datatype.hpp"
#include "gaussian.hpp"

#ifdef __x86_64__
   #include <immintrin.h>
#else   
   #include "sse2neon.h"
#endif

//#define IM2COL 0
#define VECTOR_ALIGNEMENT 64
//#define BLOCK_SIZE 256

#define TILE_M 4 // 4 ops
#define TILE_N 16 // AVX2 = 2 ops * 8 floats
#define TILE_K 16 // loop
#define PUT_IN_REGISTER register

using namespace std;
using namespace cv;

SRCNN::SRCNN()
{ 
    for(unsigned int i = 0; i < this->weights.size(); i++)
    {
        this->weights[i] = this->weights[i].insert(0, this->basePath);
    }
}

void SRCNN::generate(string filename)
{
    this->img = imread(filename, IMREAD_COLOR);
    cvtColor(this->img, this->gray, COLOR_BGR2GRAY); // my SRCNN can only accept grayscale image

    // downsample to half size of width and height
    resize(this->gray, this->downsample, Size(), 1.0 / this->scale, 1.0 / this->scale, INTER_CUBIC);

    // using bicubic to resize input image
    Mat bicubicTemp;
    resize(this->downsample, this->bicubic, Size(), this->scale, this->scale, INTER_CUBIC);
    //this->bicubic = bicubicTemp;

    // prepare input, output, and conv data
    cout << "prepare data" << endl;
    int inputWidth = this->bicubic.cols;
    int inputHeight = this->bicubic.rows;
    cout << "input width height " << inputWidth << " " << inputHeight << endl;
    ImageDim inputDim = make_tuple(1, inputHeight, inputWidth);
    data_t *input = new data_t[inputHeight * inputWidth];
    ImageDim conv1Dim = make_tuple(64, inputHeight, inputWidth);
    ImageDim conv2Dim = make_tuple(32, inputHeight, inputWidth);
    ImageDim conv3Dim = make_tuple(1, inputHeight, inputWidth);
    data_t *conv1Data = new data_t[getTotalDimension(conv1Dim)];
    data_t *conv2Data = new data_t[getTotalDimension(conv2Dim)];
    data_t *conv3Data = new data_t[getTotalDimension(conv3Dim)];
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;
    cout << "output width height " << outputWidth << " " << outputHeight << endl;
    data_t *dst = new data_t[outputHeight * outputWidth];
    cout << "assign input and output value" << endl;
#pragma omp parallel for
    for(int i = 0; i < inputHeight; i++)
    {
        for(int j = 0; j < inputWidth; j++)
        {
            input[(i * inputWidth) + j] = (data_t)this->bicubic.at<uchar>(i, j) / 255.0;
            dst[(i * inputWidth) + j] = 0;
        }
    }

    // read conv and bias weights
    cout << "read conv and bias weights" << endl;
    cout << "kernelDim" << endl;
    KernelDim conv1WeightsDim = make_tuple(64, 1, 9, 9);
    KernelDim conv2WeightsDim = make_tuple(32, 64, 5, 5);
    KernelDim conv3WeightsDim = make_tuple(1, 32, 5, 5);
    cout << "biasDim" << endl;
    ImageDim bias1Dim = make_tuple(64, 1, 1);
    ImageDim bias2Dim = make_tuple(32, 1, 1);
    ImageDim bias3Dim = make_tuple(1, 1, 1);
    cout << "finish setting bias dim" << endl;
    data_t *conv1Weights = new data_t[getTotalDimension(conv1WeightsDim)];
    data_t *conv1Weights_transposed = new data_t[getTotalDimension(conv1WeightsDim)];
    data_t *conv2Weights = new data_t[getTotalDimension(conv2WeightsDim)];
    data_t *conv2Weights_transposed = new data_t[getTotalDimension(conv2WeightsDim)];
    data_t *conv3Weights = new data_t[getTotalDimension(conv3WeightsDim)];
    data_t *conv3Weights_transposed = new data_t[getTotalDimension(conv3WeightsDim)];
    data_t *bias1Weights = new data_t[getTotalDimension(bias1Dim)];
    data_t *bias2Weights = new data_t[getTotalDimension(bias2Dim)];
    data_t *bias3Weights = new data_t[getTotalDimension(bias3Dim)]; 
    cout << "finish allocating conv and bias weights' space" << endl; 
#if IM2COL
    int conv1_col_height = get<2>(conv1WeightsDim) * get<3>(conv1WeightsDim) * get<1>(conv1WeightsDim);
    int conv1_col_width = get<1>(conv1Dim) * get<2>(conv1Dim);
    data_t *conv1Workspace = new data_t[conv1_col_height * conv1_col_width];
    int conv2_col_height = get<2>(conv2WeightsDim) * get<3>(conv2WeightsDim) * get<1>(conv2WeightsDim);
    int conv2_col_width = get<1>(conv2Dim) * get<2>(conv2Dim);
    data_t *conv2Workspace = new data_t[conv2_col_height * conv2_col_width];
    int conv3_col_height = get<2>(conv3WeightsDim) * get<3>(conv3WeightsDim) * get<1>(conv3WeightsDim);
    int conv3_col_width = get<1>(conv3Dim) * get<2>(conv3Dim);
    data_t *conv3Workspace = new data_t[conv3_col_height * conv3_col_width];
#endif

    readConvWeights(this->weights[0], conv1Weights, conv1WeightsDim, NCWH, false); cout << "weight[0]" << endl;
    readConvWeights(this->weights[1], conv2Weights, conv2WeightsDim, CHWN, true); cout << "weight[1]" << endl;
    transpose(conv2Weights_transposed, conv2Weights, 64 * 5 * 5, 32);// CHWN -> NCHW
    readConvWeights(this->weights[2], conv3Weights, conv3WeightsDim, CHWN, false); cout << "weights[2]" << endl;
    readBiasWeights(this->weights[3], bias1Weights); cout << "weight[3]" << endl;
    readBiasWeights(this->weights[4], bias2Weights); cout << "weight[4]" << endl;
    readBiasWeights(this->weights[5], bias3Weights); cout << "weight[5]" << endl;

    auto start = chrono::steady_clock::now();

    // conv1 (feature extraction)
    //cout << "conv1" << endl;
#if IM2COL
    convolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim, conv1Workspace);
#else
    naiveConvolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim);
#endif
    activation(conv1Data, conv1Data, conv1Dim, RELU);
#if 0 
    data_t *conv1arr = new data_t[get<1>(conv1Dim) * get<2>(conv1Dim)];
    for(int i = 0; i < get<0>(conv1Dim); i++)
    {
        for(int j = 0; j < get<1>(conv1Dim); j++)
        {
            for(int k = 0; k < get<2>(conv1Dim); k++)
            {
                conv1arr[j * get<2>(conv1Dim) + k] = conv1Data[(i * get<1>(conv1Dim) + j) * get<2>(conv1Dim) + k];
            }
        }
        Mat conv1(get<1>(conv1Dim), get<2>(conv1Dim), CV_32FC1, conv1arr);
        conv1.convertTo(conv1, CV_8UC1, 255.0);
        string outputname = "conv1_" + to_string(i) + ".jpg";
        imwrite(outputname, conv1);
    }
    delete [] conv1arr;
#endif

    // conv2 (non-linear mapping)
    //cout << "conv2" << endl;
#if IM2COL
    convolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights_transposed, conv2WeightsDim, 1, bias2Weights, bias2Dim, conv2Workspace);
#else
    naiveConvolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights_transposed, conv2WeightsDim, 1, bias2Weights, bias2Dim);
#endif
    activation(conv2Data, conv2Data, conv2Dim, RELU);
#if 0
    data_t *conv2arr = new data_t[get<1>(conv2Dim) * get<2>(conv2Dim)];
    for(int i = 0; i < 32; i++)
    {
        for(int j = 0; j < get<1>(conv2Dim); j++)
        {
            for(int k = 0; k < get<2>(conv2Dim); k++)
            {
                conv2arr[j * get<2>(conv2Dim) + k] = conv2Data[(i * get<1>(conv2Dim) + j) * get<2>(conv2Dim) + k];
            }
        }
        Mat conv2(get<1>(conv2Dim), get<2>(conv2Dim), CV_32FC1, conv2arr);
        conv2.convertTo(conv2, CV_8UC1, 255.0);
        string outputname = "conv2_" + to_string(i) + ".jpg";
        imwrite(outputname, conv2);
    }
    delete [] conv2arr;
#endif

    // conv3 (reconstruction)
    //cout << "conv3" << endl;
#if IM2COL
    convolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim, conv3Workspace);
#else
    naiveConvolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim);
#endif
#if 0
    unsigned char *conv3arr = new unsigned char[get<1>(conv3Dim) * get<2>(conv3Dim)];
    for(int i = 0; i < get<0>(conv3Dim); i++)
    {
        for(int j = 0; j < get<1>(conv3Dim); j++)
        {
            for(int k = 0; k < get<2>(conv3Dim); k++)
            {
                conv3arr[j * get<2>(conv3Dim) + k] = (unsigned char) (conv3Data[(i * get<1>(conv3Dim) + j) *
                                                     get<2>(conv3Dim) + k] * 255.0f);
                if(conv3arr[j * get<2>(conv3Dim) + k] > 255) conv3arr[j * get<2>(conv3Dim) + k] = 255;
                //cout << (int)conv3arr[j * get<2>(conv3Dim) + k] << endl;
                //cout << conv3Data[(i * get<1>(conv3Dim) + j) * get<2>(conv3Dim) + k] << endl;
            }
        }
        Mat conv3(get<1>(conv3Dim), get<2>(conv3Dim), CV_8UC1, conv3arr);
        //conv3.convertTo(conv3, CV_8UC1, 255.0);
        string outputname = "conv3_" + to_string(i) + ".jpg";
        imwrite(outputname, conv3);
    }
    //delete [] conv3arr;
#endif

    auto end = chrono::steady_clock::now();
    auto diff = end - start;
    cout << chrono::duration<data_t>(diff).count() << " s" << endl;

    cout << "prepare output" << endl;
#pragma omp parallel for
    for(int i = 0; i < outputHeight; i++)
    {
        for(int j = 0; j < outputWidth; j++)
        {
            //cout << i << " " << j << " fine" << endl;
            dst[(i * outputWidth) + j] = conv3Data[((1 - 1) * get<1>(conv3Dim) + i) * get<2>(conv3Dim) + j];
            //cout << dst[(i * outputWidth) + j] << endl;
            //dst[(i * outputWidth) + j] = conv3Data[(i * outputWidth) + j];
#if 0
            if(dst[(i * outputWidth) + j] != 0)
            {
                cout << "index " << i << " " << j << " " << conv3Data[(i * outputWidth) + j] << endl;
            }
#endif
        }
    }

    // copy to output OpenCV Mat
    cout << "copy to output OpenCV Mat" << endl;
    #if USEFLOAT
    Mat SRCNN(outputHeight, outputWidth, CV_32FC1, dst);
    #else
    Mat SRCNN(outputHeight, outputWidth, CV_64FC1, dst);
    #endif
    SRCNN.convertTo(SRCNN, CV_8UC1, 255);
    this->output = SRCNN;


    delete [] input;
    delete [] conv1Data;
    delete [] conv2Data;
    delete [] conv3Data;
    delete [] conv1Weights;
    delete [] conv1Weights_transposed;
    delete [] conv2Weights;
    delete [] conv2Weights_transposed;
    delete [] conv3Weights;
    delete [] conv3Weights_transposed;
    delete [] bias1Weights;
    delete [] bias2Weights;
    delete [] bias3Weights;

#if IM2COL
    delete [] conv1Workspace;
    delete [] conv2Workspace;
    delete [] conv3Workspace;
#endif
}

void SRCNN::showOutput()
{
    namedWindow("input");
    imshow("input", this->img);
    waitKey(0);
    namedWindow("downsample");
    imshow("downsample", this->downsample);
    waitKey(0);
    namedWindow("bicubic");
    imshow("bicubic", this->bicubic);
    waitKey(0);
    namedWindow("SRCNN");
    imshow("SRCNN", this->output);
    waitKey(0);
}

void SRCNN::outputImage()
{
    imwrite("srcnnResult.bmp", this->output);
}

int SRCNN::getTotalDimension(ImageDim dim)
{
    return get<0>(dim) * get<1>(dim) * get<2>(dim);
}

int SRCNN::getTotalDimension(KernelDim dim)
{
    return get<0>(dim) * get<1>(dim) * get<2>(dim) * get<3>(dim);
}

void SRCNN::checkWeightStatus()
{
    for(string weightPath: this->weights)
    {
        ifstream input(weightPath);
        if(!input.is_open())
        {
            cerr << "file " << weightPath << " opened unsuccessfully" << endl;
            continue;
        }
        vector<data_t> contents;
        data_t temp;
        while(input >> temp)
        {
            contents.push_back(temp);
        }
        
        cout << weightPath << " " << contents.size() << "\t";
        for(unsigned int i = 0; i < 3; i++)
        {
            cout << contents[i] << " ";
        }
        cout << endl;
        input.close();
    }
}

void SRCNN::testImageConv(string filename)
{
    Mat gray = imread(filename, IMREAD_GRAYSCALE);
    imshow("gray", gray);
    waitKey(0);

    // ordinary Gaussian filter
    int width = gray.cols;
    int height = gray.rows;
    unsigned char *source = new unsigned char[height * width];
    unsigned char *destination = new unsigned char[height * width];

    // Conv test 
    data_t *input = new data_t[1 * height * width];
    data_t *output = new data_t[1 * height * width];
    ImageDim inputDim = make_tuple(1, height, width);
    ImageDim outputDim = make_tuple(1, height, width);
    unsigned char *dst = new unsigned char[height * width];

    int kernelWidth = 3;
    int kernelHeight = 3;
    data_t sigma = 3.0;

    // Conv test
    data_t *kernel = new data_t[kernelHeight * kernelWidth];
    data_t *kernel_data_t = new data_t[kernelHeight * kernelWidth];
    KernelDim kernelDim = make_tuple(1, 1, kernelHeight, kernelWidth);
#pragma omp parallel for
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            source[(i * width) + j] = gray.at<uchar>(i, j);
            destination[(i * width) + j] = source[(i * width) + j];

            input[(i * width) + j] = gray.at<uchar>(i, j) / 255.0;
            output[(i * width) + j] = input[(i * width) + j];
        }
    }
    generateKernel(kernelWidth, kernelHeight, sigma, kernel);
    for(int i = 0; i < kernelHeight; i++)
    {
        for(int j = 0; j < kernelWidth; j++)
        {
            kernel_data_t[i * kernelWidth + j] = (data_t)kernel[i * kernelWidth + j];
        }
    }

    gaussianFilter(source, destination, width, height, kernelWidth, kernelHeight, sigma);
    Mat result(height, width, CV_8UC1, destination);
    imshow("gaussian", result);
    waitKey(0);

    convolution(input, output, inputDim, outputDim, kernel_data_t, kernelDim); 
    /*testConvolution(input, output, inputDim, outputDim, kernel_data_t, kernelDim, 1, NULL, std::make_tuple(0, 0, 0),
                    "deadbeef", "deadbeef");*/
    int counter = 0;
#pragma omp parallel for
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            dst[(i * width) + j] = output[(i * width) + j] * 255.0;
            if(dst[(i * width) + j] != destination[(i * width) + j])
            {
                counter++;
            }
        }
    }
    cout << "total " << counter << " differences" << endl;
    cout << "height: " << height << " width: " << width << endl;
    Mat result1(height, width, CV_8UC1, dst);
    //Mat result1(height, width, CV_64FC1, output);
    imshow("gaussian 1", result1);
    waitKey(0);
}

void SRCNN::testConv1Channel()
{
    // input
    ImageDim inputDim = make_tuple(1, 5, 5);
    data_t input[]
    {
     1, 2, 3, 4, 5,
     6, 7, 8, 9, 10,
     11, 12, 13, 14, 15,
     16, 17, 18, 19, 20,
     21, 22, 23, 24, 25
    };

    // kernel
    KernelDim kernelDim = make_tuple(1, 1, 3, 3);
    data_t kernel[]
    {
     0, 0, 0,
     0, 1, 0,
     0, 0, 0
    };

    // output
    ImageDim outputDim = make_tuple(1, 5, 5);
    data_t *output = new data_t[getTotalDimension(outputDim)];

    // bias
    ImageDim biasDim = make_tuple(1, 1, 1);
    data_t bias[] = { 0 };

    // apply convolution
    convolution(input, output, inputDim, outputDim,
                kernel, kernelDim, 1, bias, 
                biasDim);
    /*testConvolution(input, output, inputDim, outputDim,
                    kernel, kernelDim, 1, bias, biasDim,
                    "deadbeef", "deadmilk");*/

    // print the convoluted result
    int outputHeight = get<1>(outputDim);
    int outputWidth = get<2>(outputDim);
    for(int i = 0; i < get<0>(outputDim); i++)
    {
#pragma omp parallel for
        for(int j = 0; j < outputHeight; j++)
        {
            for(int k = 0; k < outputWidth; k++)
            {
                cout << output[((i) * outputHeight + j) * outputWidth + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

// http://cs231n.github.io/convolutional-networks/
void SRCNN::testConv3Channels()
{
    // input
    ImageDim inputDim = make_tuple(3, 5, 5);
    data_t input[] = 
    {
        /* channel 0 */
        2, 1, 1, 0, 1,
        1, 1, 0, 2, 1,
        2, 0, 0, 2, 2,
        2, 0, 1, 1, 1,
        1, 2, 2, 2, 0,
        /* channel 1 */
        0, 0, 0, 2, 1,
        0, 0, 2, 0, 2,
        0, 2, 2, 2, 1,
        0, 0, 2, 0, 1,
        0, 0, 2, 1, 1,
        /* channel 2 */
        0, 2, 2, 0, 2,
        0, 1, 1, 0, 2,
        1, 2, 1, 1, 0,
        2, 2, 0, 2, 2,
        2, 0, 2, 2, 0
    };

    // output
    int outputDepth = 2;
    int outputHeight = 3;
    int outputWidth = 3;
    ImageDim outputDim = make_tuple(2, 3, 3);
    data_t *output = new data_t[getTotalDimension(outputDim)];
    for(int i = 0; i < outputDepth; i++)
    {
#pragma omp parallel for
        for(int j = 0; j < outputHeight; j++)
        {
            for(int k = 0; k < outputWidth; k++)
            {
                output[(i * outputHeight + j) * outputWidth + k] = 0;
            }
        }
    }

    // kernel
    KernelDim filtersDim = make_tuple(2, 3, 3, 3);
    data_t filters[] = 
    {
        /* filter w0 */
        /* channel 0 */
        0, 0, 1,
        0, 0, -1,
        1, 0, 0,
        /* channel 1 */
        -1, 0, 1,
        1, 1, 0,
        0, 0, 0,
        /* channel 2 */
        1, -1, -1,
        1, 0, -1,
        1, -1, 0,

        /* filter w1 */
        /* channel 0 */
        1, 1, 0,
        0, 1, 1,
        1, 1, 0,
        /* channel 1 */
        -1, 1, -1,
        -1, 0, 1,
        -1, 0, 1,
        /* channel 2 */
        1, -1, 1,
        1, 1, 0,
        -1, -1, 1
    };


    // bias
    ImageDim biasesDim = make_tuple(2, 1, 1);
    data_t biases[] = 
    {/* b0 */
     1,
     /* b1 */
     0
    };

    // operate convolution on test data
    convolution(input, output, inputDim,
                outputDim, filters, filtersDim, 2,
                biases, biasesDim);
    /*testConvolution(input, output, inputDim,
                outputDim, filters, filtersDim, 2,
                biases, biasesDim, "deadbeef", "deadmilk");*/
    // print the convoluted result
    for(int i = 0; i < get<0>(outputDim); i++)
    {
        for(int j = 0; j < get<1>(outputDim); j++)
        {
            for(int k = 0; k < get<2>(outputDim); k++)
            {
                cout << output[((i) * outputHeight + j) * outputWidth + k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

// matrix transpose test
void SRCNN::testTranspose()
{
    // kernel
    KernelDim filtersDim = make_tuple(2, 3, 3, 3);
    int kernel_num = 2;
    int kernel_c = 3;
    int kernel_h = 3;
    int kernel_w = 3;
    data_t testKernel[] = 
    {
        /* filter w0 */
        /* channel 0 */
        0, 0, 1,
        0, 0, -1,
        1, 0, 0,
        /* channel 1 */
        -1, 0, 1,
        1, 1, 0,
        0, 0, 0,
        /* channel 2 */
        1, -1, -1,
        1, 0, -1,
        1, -1, 0,

        /* filter w1 */
        /* channel 0 */
        1, 1, 0,
        0, 1, 1,
        1, 1, 0,
        /* channel 1 */
        -1, 1, -1,
        -1, 0, 1,
        -1, 0, 1,
        /* channel 2 */
        1, -1, 1,
        1, 1, 0,
        -1, -1, 1
    };

    data_t *filters_transposed = new data_t[getTotalDimension(filtersDim)];
    int filters_transposed_h = kernel_c * kernel_h * kernel_w;
    int filters_transposed_w = kernel_num;

    data_t *result = new data_t[getTotalDimension(filtersDim)];
    int result_h = kernel_num;
    int result_w = kernel_c * kernel_h * kernel_w;

    transpose(filters_transposed, testKernel, kernel_num, kernel_c * kernel_h * kernel_w);
    cout << "transpose: " << endl;
    for(int i = 0; i < filters_transposed_h; i++)
    {
        for(int j = 0; j < filters_transposed_w; j++)
        {
            cout << filters_transposed[i * filters_transposed_w + j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    transpose(result, filters_transposed, filters_transposed_h, filters_transposed_w);
    cout << "result: " << endl;
    for(int i = 0; i < kernel_num; i++)
    {
        for(int j = 0; j < kernel_c * kernel_h * kernel_w; j++)
        {
            cout << result[i * kernel_c * kernel_h * kernel_w + j] << " ";
        }
        cout << endl;
    }

    delete [] filters_transposed;
    delete [] result;
}

void SRCNN::testReadAndTranspose()
{
    KernelDim conv1WeightsDim = make_tuple(64, 1, 9, 9);
    KernelDim conv2WeightsDim = make_tuple(32, 64, 5, 5);
    KernelDim conv3WeightsDim = make_tuple(1, 32, 5, 5);
    data_t *conv1Weights = new data_t[getTotalDimension(conv1WeightsDim)];
    data_t *conv1Weights_transposed = new data_t[getTotalDimension(conv1WeightsDim)];
    data_t *conv1Weights_tt = new data_t[getTotalDimension(conv1WeightsDim)];
    data_t *conv2Weights = new data_t[getTotalDimension(conv2WeightsDim)];
    data_t *conv2Weights_transposed = new data_t[getTotalDimension(conv2WeightsDim)];
    data_t *conv2Weights_tt = new data_t[getTotalDimension(conv2WeightsDim)];
    data_t *conv3Weights = new data_t[getTotalDimension(conv3WeightsDim)];
    data_t *conv3Weights_transposed = new data_t[getTotalDimension(conv3WeightsDim)];
    data_t *conv3Weights_tt = new data_t[getTotalDimension(conv3WeightsDim)];
    
    readConvWeights(this->weights[0], conv1Weights); cout << "weight[0]" << endl;
    readConvWeights(this->weights[1], conv2Weights, true); cout << "weight[1]" << endl;
    readConvWeights(this->weights[2], conv3Weights, false, true); cout << "weight[2]" << endl;

    int conv1_row = 64, conv1_col = 1*9*9;
    transpose(conv1Weights_transposed, conv1Weights, conv1_row, conv1_col);
    transpose(conv1Weights_tt, conv1Weights_transposed, conv1_col, conv1_row);
    for(int i = 0; i < conv1_row; i++)
    {
        for(int j = 0; j < conv1_col; j++)
        {
            if(conv1Weights_tt[i * conv1_col + j] != conv1Weights[i * conv1_col + j])
            {
                cout << "index " << i * conv1_col + j << " not same after transpose and transpose" << endl; 
            }
        }
    }
}

void SRCNN::testReadWeightFormat()
{
    KernelDim conv1Dim = make_tuple(64, 1, 9, 9);
    KernelDim conv2Dim = make_tuple(32, 64, 5, 5);
    KernelDim conv3Dim = make_tuple(1, 32, 5, 5);
    data_t *conv1Weight = new data_t[getTotalDimension(conv1Dim)];
    data_t *conv2Weight = new data_t[getTotalDimension(conv2Dim)];
    data_t *conv2Weight_transposed = new data_t[getTotalDimension(conv2Dim)];
    data_t *conv3Weight = new data_t[getTotalDimension(conv3Dim)];
    readConvWeights(this->weights[0], conv1Weight, conv1Dim, NCWH);
    readConvWeights(this->weights[1], conv2Weight, conv2Dim, CHWN, true);
    readConvWeights(this->weights[2], conv3Weight, conv3Dim, CHWN);

    transpose(conv2Weight_transposed, conv2Weight, 64*5*5, 32);

    // weight format write test
    testWriteWeights("myWeightConv1Dump", conv1Weight, conv1Dim);
    testWriteWeights("myWeightConv2Dump", conv2Weight_transposed, conv2Dim);
    testWriteWeights("myWeightConv3Dump", conv3Weight, conv3Dim);

    delete [] conv1Weight;
    delete [] conv2Weight;
    delete [] conv2Weight_transposed;
    delete [] conv3Weight;
}

void SRCNN::naiveConvolution(data_t *input, data_t *output, ImageDim inputDim,
    ImageDim outputDim, data_t *kernels, KernelDim kernelDim, int stride/* = 1*/,
    data_t *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/)
{
    int kernelOutputChannel = get<0>(kernelDim);
    int kernelInputChannel = get<1>(kernelDim);
    int kernelHeight = get<2>(kernelDim);
    int kernelWidth = get<3>(kernelDim);
    int kernelHeightSize = kernelHeight / 2;
    int kernelWidthSize = kernelWidth / 2;

    int inputChannel = get<0>(inputDim);
    int inputHeight = get<1>(inputDim);
    int inputWidth = get<2>(inputDim);

    int outputChannel = get<0>(outputDim);
    int outputHeight = get<1>(outputDim);
    int outputWidth = get<2>(outputDim); 

    for(int k = 0; k < outputChannel; k++)
    {
        for(int n = 0; n < inputChannel; n++)
        {
#pragma omp parallel for
            for(int i = 0; i < inputHeight; i += stride)
            {
                for(int j = 0; j < inputWidth; j += stride)
                {
                    data_t sum = 0.0;
                    for(int l = -kernelHeightSize; l <= kernelHeightSize; l++)
                    {
                        for(int m = -kernelWidthSize; m <= kernelWidthSize; m++)
                        {
                            int y = i + l;
                            int x = j + m;

                            // valid padding
                            x = x >= 0 ? (x < inputWidth ? x : inputWidth - stride) : 0;
                            y = y >= 0 ? (y < inputHeight ? y : inputHeight - stride) : 0;

                            int inputIdx = (n * inputHeight * inputWidth) + (y * inputWidth) + x;

                            int kernelIdx = ((k * kernelInputChannel + n) 
                                            * kernelHeight + (l + kernelHeightSize))
                                            * kernelWidth + (m + kernelWidthSize);

                            sum += input[inputIdx] * kernels[kernelIdx]; 
                        }
                    }

                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += sum;
                }
            }
        }

        if(bias != NULL)
        {
#pragma omp parallel for
            for(int i = 0; i < outputHeight; i++)
            {
                for(int j = 0; j < outputWidth; j++)
                {
                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += bias[k];
                }
            }
        }
    }
}

// standard convolution
void SRCNN::convolution(data_t *input, data_t *output, ImageDim inputDim,
    ImageDim outputDim, data_t *kernels, KernelDim kernelDim, int stride/* = 1*/,
    data_t *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/,
    data_t *workspace /*= NULL*/)
{
    int kernelOutputChannel = get<0>(kernelDim);
    int kernelInputChannel = get<1>(kernelDim);
    int kernelHeight = get<2>(kernelDim);
    int kernelWidth = get<3>(kernelDim);
    int kernelHeightSize = kernelHeight / 2;
    int kernelWidthSize = kernelWidth / 2;

    int inputChannel = get<0>(inputDim);
    int inputHeight = get<1>(inputDim);
    int inputWidth = get<2>(inputDim);

    int outputChannel = get<0>(outputDim);
    int outputHeight = get<1>(outputDim);
    int outputWidth = get<2>(outputDim);

    /* Add assertion here in the future */

    // input dimension = C * H * W = (K * K * C) * N
    // where N = out_h * out_w
    data_t *input_col;
    int input_col_height = kernelHeight * kernelWidth * inputChannel;
    int input_col_width = outputHeight * outputWidth;
    if(!workspace)
    {
        input_col = new data_t[input_col_height * input_col_width];
    } else
    {
        input_col = workspace;
    }

    int padding = kernelHeightSize; // temporary setting, may be changed in the future

    im2col(input, inputDim, kernelDim, stride, padding, input_col);

    // Output input_col
#if 0
    int line_counter = 0;
    cout << "input_col content:" << endl;
    for(int i = 0; i < input_col_height * input_col_width; i++)
    {
        cout << setw(2) << input_col[i] << " ";
        line_counter++;
        if(line_counter % (outputHeight * outputWidth) == 0)
        {
            cout << endl;
        }
    }
#endif

    // Output kernel
    // Note the dimension is changed from NCHW to Nx(CxKxK)
    int kernel_col_height = kernelOutputChannel;
    int kernel_col_width = kernelInputChannel * kernelHeight * kernelWidth;
#if 0
    line_counter = 0;
    cout << endl << "kernel content:" << endl;
    for(int i = 0; i < kernel_col_height; i++)
    {
        for(int j = 0; j < kernel_col_width; j++)
        {
            cout << setw(2) << kernels[i * kernel_col_width + j] << " ";
            line_counter++;
            if(line_counter % kernel_col_width == 0)
            {
                cout << endl;
            }
        }
    }
#endif

    matMul(output, kernels, input_col, bias,
           kernel_col_height, kernel_col_width, input_col_height, input_col_width);

    if(!workspace)
    {
        delete [] input_col;
    }
}


/* From Berkeley Vision's Caffe!
 * https://github.com/BVLC/caffe/blob/master/LICENSE
 */
void SRCNN::im2col(data_t *data_im, ImageDim imageDim, KernelDim kernelDim,
                   int stride, int pad, data_t *data_col)
{
    int imageHeight = get<1>(imageDim);
    int imageWidth = get<2>(imageDim);
    int kernelHeight = get<2>(kernelDim);
    int kernelWidth = get<3>(kernelDim);
    int col_height = (imageHeight + 2 * pad - kernelHeight) / stride + 1;
    int col_width = (imageWidth + 2 * pad - kernelWidth) / stride + 1;
    int imageChannel = get<0>(imageDim);
    int col_channel = imageChannel * kernelHeight * kernelWidth;

    for(int c = 0; c < col_channel; c++)
    {
        int w_offset = c % kernelWidth;
        int h_offset = (c / kernelWidth) % kernelHeight;
        int c_im = c / kernelWidth / kernelHeight;
//#pragma omp parallel for
        for(int h = 0; h < col_height; h++)
        {
            for(int w = 0; w < col_width; w++)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_idx = (c * col_height + h) * col_width + w;
                data_col[col_idx] = im2colGetPixel(data_im, imageDim, 
                                                   im_row, im_col, c_im, pad);
            }
        }
    }
}

void SRCNN::col2im(data_t *data_col, ImageDim imageDim, KernelDim kernelDim,
                int stride, int pad, data_t *data_im)
{
    int imageHeight = get<1>(imageDim);
    int imageWidth = get<2>(imageDim);
    int kernelHeight = get<2>(kernelDim);
    int kernelWidth = get<3>(kernelDim);
    int col_height = (imageHeight + 2 * pad - kernelHeight) / stride + 1;
    int col_width = (imageWidth + 2 * pad - kernelWidth) / stride + 1;
    int imageChannel = get<0>(imageDim);
    int col_channel = imageChannel * kernelHeight * kernelWidth;

    for(int c = 0; c < col_channel; c++)
    {
        int w_offset = c % kernelWidth;
        int h_offset = (c / kernelWidth) % kernelHeight;
        int c_im = c / kernelWidth / kernelHeight;
//#pragma omp parallel for
        for(int h = 0; h < col_height; h++)
        {
            for(int w = 0; w < col_width; w++)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_idx = (c * col_height + h) * col_width + w;
                data_t value = data_col[col_idx];
                col2imAddPixel(data_im, imageDim, im_row, im_col,
                               c_im, pad, value);
            }
        }
    }
}

data_t SRCNN::im2colGetPixel(data_t *im, ImageDim imageDim, 
                             int row, int col, int channel, int pad)
{
    int height = get<1>(imageDim);
    int width = get<2>(imageDim);

    row -= pad;
    col -= pad;

    // zero padding
#if 0
    if(row < 0 || col < 0 || row >= height || col >= width)
    {
        return 0;
    }
#endif
    // reflect padding
//#if 0
    if(row < 0) row = 0;
    if(col < 0) col = 0;
    if(row >= height) row = height - 1;
    if(col >= width) col = width - 1;
//#endif

    return im[col + width * (row + height * channel)];
}

void SRCNN::col2imAddPixel(data_t *im, ImageDim imageDim,
                           int row, int col, int channel, int pad, data_t value)
{   
    int height = get<1>(imageDim);
    int width = get<2>(imageDim);

    row -= pad;
    col -= pad;

    // zero padding
    if(row < 0 || col < 0 || row >= height || col >= width)
    {
        return;
    }

    im[col + width * (row + height * channel)] += value;
}

void SRCNN::matMul(data_t *out, data_t *kernel, data_t *in, data_t *bias,
                   int kernel_row, int kernel_col, int in_row, int in_col)
{
    if(bias == NULL)
    {
        naiveGEMM(out, kernel, in, 
                  kernel_row, kernel_col, in_row, in_col);
    }
    else
    {
        /*naiveGEMM_addBias(out, kernel, in, bias,
                          kernel_row, kernel_col, in_row, in_col);*/
        #ifdef ISX86
        /*intrinsicGEMM_addBias(out, kernel, in, bias,
                          kernel_row, kernel_col, in_row, in_col);*/
        /*intrinsicGEMM_microkernel_addBias(out, kernel, in, bias,
                          kernel_row, kernel_col, in_row, in_col);*/
        intrinsicGEMM_microkernel_with_packing_addBias(out, kernel, in, bias,
                          kernel_row, kernel_col, in_row, in_col);

        #else
        tiledNVectorizedGEMM_addBias(out, kernel, in, bias,
                          kernel_row, kernel_col, in_row, in_col);
        #endif
    }
}
 
void SRCNN::naiveGEMM(data_t *out, data_t *kernel, data_t *in,
                      int kernel_row, int kernel_col, int in_row, int in_col)
{
     /* The output matrix dimension will be kernel_row * in_col */
    assert(kernel_col == in_row);

    memset(out, 0, sizeof(data_t) * kernel_row * in_col);
#pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        //for(int j = 0; j < in_col; j++)
        for(int k = 0; k < in_row; k++)
        {
            //out[i * in_col + j] = 0;
            //for(int k = 0; k < in_row; k++)
            for(int j = 0; j < in_col; j++)
            {
                out[i * in_col + j] +=
                    kernel[i * kernel_col + k] *
                    in[k * in_col + j];
            }
        }
    }
}

void SRCNN::naiveGEMM_addBias(data_t *out, data_t *kernel, data_t *in, data_t *bias,
                              int kernel_row, int kernel_col, int in_row, int in_col)
{
    /* The output matrix dimension will be kernel_row * in_col */
    assert(kernel_col == in_row);

    memset(out, 0, sizeof(data_t) * kernel_row * in_col);

    #pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        //for(int k = 0; k < in_row; k++)
        {
            //out[i * in_col + j] = 0;
            for(int k = 0; k < in_row; k++)
            //for(int j = 0; j < in_col; j++)
            {
                out[i * in_col + j] +=
                    kernel[i * kernel_col + k] *
                    in[k * in_col + j];
            }
        }
    }

#pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] += bias[i];
        }
    }
        
}

void SRCNN::tiledNVectorizedGEMM_addBias(data_t * __restrict__ pout, data_t * __restrict__ pkernel, data_t * __restrict__ pin, data_t *bias,
                              int kernel_row, int kernel_col, int in_row, int in_col)
{
    /* The output matrix dimension will be kernel_row * in_col */
    assert(kernel_col == in_row);

    const data_t *kernel = (const data_t *)__builtin_assume_aligned(pkernel, VECTOR_ALIGNEMENT);
    const data_t *in = (const data_t *)__builtin_assume_aligned(pin, VECTOR_ALIGNEMENT);
    data_t *out = (data_t *)__builtin_assume_aligned(pout, VECTOR_ALIGNEMENT);

    memset(out, 0, sizeof(data_t) * kernel_row * in_col);

    for(int ii = 0; ii < kernel_row; ii += BLOCK_SIZE_X)
    {
        for(int kk = 0; kk < in_row; kk += BLOCK_SIZE_Z)
        {
            for(int jj = 0; jj < in_col; jj += BLOCK_SIZE_Y)
            {
                int maxi = min(ii + BLOCK_SIZE_X, kernel_row);
                int maxk = min(kk + BLOCK_SIZE_Z, in_row);
                int maxj = min(jj + BLOCK_SIZE_Y, in_col);
                #pragma omp parallel for
                for(int i = ii; i < maxi; i++)
                {
                    for(int k = kk; k < maxk; k++)
                    {
                        data_t temp = kernel[i * kernel_col + k];
                        for(int j = jj; j < maxj; j++)
                        {
                            out[i * in_col + j] +=
                                temp *
                                in[k * in_col + j];
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] += bias[i];
        }
    }
        
}

#ifdef ISX86
void SRCNN::intrinsicGEMM_addBias(float *out, float *kernel, float *in, float *bias,
                      int kernel_row, int kernel_col, int in_row, int in_col)
{
    /* The output matrix dimension will be kernel_row * in_col */
    assert(kernel_col == in_row);

    memset(out, 0, sizeof(float) * kernel_row * in_col);

    /*#pragma omp parallel for
    for(int t = 0; t < kernel_row; t++)
        gemm_nn(1, in_col, kernel_col, 1.0, 
                kernel + t * kernel_col, kernel_col,
                in, in_col,
                out + t * in_col, in_col);*/
    for(int ii = 0; ii < kernel_row; ii += BLOCK_SIZE_X)
    {
        for(int kk = 0; kk < in_row; kk += BLOCK_SIZE_Z)
        {
            for(int jj = 0; jj < in_col; jj += BLOCK_SIZE_Y)
            {
                int maxi = min(ii + BLOCK_SIZE_X, kernel_row);
                int maxk = min(kk + BLOCK_SIZE_Z, in_row);
                int maxj = min(jj + BLOCK_SIZE_Y, in_col);
                /*#pragma omp parallel for
                for(int i = ii; i < maxi; i++)
                {
                    for(int k = kk; k < maxk; k++)
                    {
                        data_t temp = kernel[i * kernel_col + k];
                        for(int j = jj; j < maxj; j++)
                        {
                            out[i * in_col + j] +=
                                temp *
                                in[k * in_col + j];
                        }
                    }
                }*/
                #pragma omp parallel for
                for(int i = ii; i < maxi; i++)
                {
                    for(int k = kk; k < maxk; k++)
                    {
                        float temp = kernel[i * kernel_col + k];
                        __m256 temp256, in256, out256, result256;
                        temp256 = _mm256_set1_ps(temp);
                        for(int j = jj; j < maxj - 8; j += 8)
                        {
                            in256 = _mm256_loadu_ps(&in[k * in_col + j]);
                            out256 = _mm256_loadu_ps(&out[i * in_col + j]);
                            // FMA
                            result256 = _mm256_fmadd_ps(temp256, in256, out256);
                            /*result256 = _mm256_mul_ps(temp256, in256);
                            result256 = _mm256_add_ps(result256, out256);*/
                            _mm256_storeu_ps(&out[i * in_col + j], result256);
                        }

                        int prev_end = (maxj % 8 == 0) ? (maxj - 8) : (maxj / 8) * 8;
                        for(int j = prev_end; j < maxj; j++)
                            out[i * in_col + j] += temp * in[k * in_col + j];
                    }
                }
            }
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] += bias[i];
        }
    }
}

void SRCNN::gemm_nn(int M, int N, int K, float ALPHA,
                    float *A, int lda,
                    float *B, int ldb,
                    float *C, int ldc)
{
    for(int i = 0; i < M; i++)
    {
        for(int k = 0; k < K; k++)
        {
            float A_PART = ALPHA * A[i * lda + k];
            __m256 a256, b256, c256, result256;
            a256 = _mm256_set1_ps(A_PART);
            for(int j = 0; j < N - 8; j += 8)
            {
                b256 = _mm256_loadu_ps(&B[k * ldb + j]);
                c256 = _mm256_loadu_ps(&C[i * ldc + j]);
                result256 = _mm256_mul_ps(a256, b256);
                result256 = _mm256_add_ps(result256, c256);
                _mm256_storeu_ps(&C[i * ldc + j], result256);
            }

            int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;
            for(int j = prev_end; j < N; j++)
                C[i * ldc + j] += A_PART * B[k * ldb + j];
        }
    }
    /*for(int i = 0; i < M; i++)
    {
        for(int k = 0; k < K; k++)
        {
            float A_PART = ALPHA * A[i * lda + k];
            for(int j = 0; j < N; j++)
            {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }*/
}

// from https://github.com/AlexeyAB/darknet/blob/master/src/gemm.c#L779
void SRCNN::intrinsicGEMM_microkernel_addBias(float *out, float *kernel, float *in, float *bias,
                      int kernel_row, int kernel_col, int in_row, int in_col)
{
    int M = kernel_row, N = in_col, K = kernel_col;
    float *A = kernel, *B = in, *C = out;
    int lda = kernel_col, ldb = in_col, ldc = in_col;
    float ALPHA = 1.0f;

    int i;

    #pragma omp parallel for
    for (i = 0; i < (M / TILE_M)*TILE_M; i += TILE_M)
    {
        int j, k;
        int i_d, k_d;

        for (k = 0; k < (K / TILE_K)*TILE_K; k += TILE_K)
        {
            for (j = 0; j < (N / TILE_N)*TILE_N; j += TILE_N)
            {
                // L1 - 6 bits tag [11:6] - cache size 32 KB, conflict for each 4 KB
                // L2 - 9 bits tag [14:6] - cache size 256 KB, conflict for each 32 KB
                // L3 - 13 bits tag [18:6] - cache size 8 MB, conflict for each 512 KB

                __m256 result256;
                __m256 a256_0, b256_0;    // AVX
                __m256 a256_1, b256_1;    // AVX
                __m256 a256_2;// , b256_2;    // AVX
                __m256 a256_3;// , b256_3;    // AVX
                __m256 c256_0, c256_1, c256_2, c256_3;
                __m256 c256_4, c256_5, c256_6, c256_7;

                c256_0 = _mm256_loadu_ps(&C[(0 + i)*ldc + (0 + j)]);
                c256_1 = _mm256_loadu_ps(&C[(1 + i)*ldc + (0 + j)]);
                c256_2 = _mm256_loadu_ps(&C[(0 + i)*ldc + (8 + j)]);
                c256_3 = _mm256_loadu_ps(&C[(1 + i)*ldc + (8 + j)]);

                c256_4 = _mm256_loadu_ps(&C[(2 + i)*ldc + (0 + j)]);
                c256_5 = _mm256_loadu_ps(&C[(3 + i)*ldc + (0 + j)]);
                c256_6 = _mm256_loadu_ps(&C[(2 + i)*ldc + (8 + j)]);
                c256_7 = _mm256_loadu_ps(&C[(3 + i)*ldc + (8 + j)]);


                for (k_d = 0; k_d < (TILE_K); ++k_d)
                {
                    a256_0 = _mm256_set1_ps(ALPHA*A[(0 + i)*lda + (k_d + k)]);
                    a256_1 = _mm256_set1_ps(ALPHA*A[(1 + i)*lda + (k_d + k)]);

                    a256_2 = _mm256_set1_ps(ALPHA*A[(2 + i)*lda + (k_d + k)]);
                    a256_3 = _mm256_set1_ps(ALPHA*A[(3 + i)*lda + (k_d + k)]);


                    b256_0 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (0 + j)]);
                    b256_1 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (8 + j)]);

                    // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                    //c256_0 = _mm256_fmadd_ps(a256_0, b256_0, c256_0);
                    //c256_1 = _mm256_fmadd_ps(a256_1, b256_0, c256_1);
                    //c256_2 = _mm256_fmadd_ps(a256_0, b256_1, c256_2);
                    //c256_3 = _mm256_fmadd_ps(a256_1, b256_1, c256_3);

                    //c256_4 = _mm256_fmadd_ps(a256_2, b256_0, c256_4);
                    //c256_5 = _mm256_fmadd_ps(a256_3, b256_0, c256_5);
                    //c256_6 = _mm256_fmadd_ps(a256_2, b256_1, c256_6);
                    //c256_7 = _mm256_fmadd_ps(a256_3, b256_1, c256_7);

                    result256 = _mm256_mul_ps(a256_0, b256_0);
                    c256_0 = _mm256_add_ps(result256, c256_0);

                    result256 = _mm256_mul_ps(a256_1, b256_0);
                    c256_1 = _mm256_add_ps(result256, c256_1);

                    result256 = _mm256_mul_ps(a256_0, b256_1);
                    c256_2 = _mm256_add_ps(result256, c256_2);

                    result256 = _mm256_mul_ps(a256_1, b256_1);
                    c256_3 = _mm256_add_ps(result256, c256_3);


                    result256 = _mm256_mul_ps(a256_2, b256_0);
                    c256_4 = _mm256_add_ps(result256, c256_4);

                    result256 = _mm256_mul_ps(a256_3, b256_0);
                    c256_5 = _mm256_add_ps(result256, c256_5);

                    result256 = _mm256_mul_ps(a256_2, b256_1);
                    c256_6 = _mm256_add_ps(result256, c256_6);

                    result256 = _mm256_mul_ps(a256_3, b256_1);
                    c256_7 = _mm256_add_ps(result256, c256_7);
                }
                _mm256_storeu_ps(&C[(0 + i)*ldc + (0 + j)], c256_0);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (0 + j)], c256_1);
                _mm256_storeu_ps(&C[(0 + i)*ldc + (8 + j)], c256_2);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (8 + j)], c256_3);

                _mm256_storeu_ps(&C[(2 + i)*ldc + (0 + j)], c256_4);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (0 + j)], c256_5);
                _mm256_storeu_ps(&C[(2 + i)*ldc + (8 + j)], c256_6);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (8 + j)], c256_7);
            }

            for (j = (N / TILE_N)*TILE_N; j < N; ++j) {
                for (i_d = i; i_d < (i + TILE_M); ++i_d)
                {
                    for (k_d = k; k_d < (k + TILE_K); ++k_d)
                    {
                        PUT_IN_REGISTER float A_PART = ALPHA*A[i_d*lda + k_d];
                        C[i_d*ldc + j] += A_PART*B[k_d*ldb + j];
                    }
                }
            }
        }

        for (k = (K / TILE_K)*TILE_K; k < K; ++k)
        {
            for (i_d = i; i_d < (i + TILE_M); ++i_d)
            {
                PUT_IN_REGISTER float A_PART = ALPHA*A[i_d*lda + k];
                for (j = 0; j < N; ++j) {
                    C[i_d*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
    }

    for (i = (M / TILE_M)*TILE_M; i < M; ++i) {
        int j, k;
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }


    #pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] += bias[i];
        }
    }
}

void SRCNN::intrinsicGEMM_microkernel_with_packing_addBias(float *out, float *kernel, float *in, float *bias,
                      int kernel_row, int kernel_col, int in_row, int in_col) {
{
    int M = kernel_row, N = in_col, K = kernel_col;
    float *A = kernel, *B = in, *C = out;
    int lda = kernel_col, ldb = in_col, ldc = in_col;
    float ALPHA = 1.0f;

    int i;

    #pragma omp parallel for
    for (i = 0; i < (M / TILE_M)*TILE_M; i += TILE_M)
    {
        int j, k;
        int i_d, k_d;

        for (k = 0; k < (K / TILE_K)*TILE_K; k += TILE_K)
        {
            for (j = 0; j < (N / TILE_N)*TILE_N; j += TILE_N)
            {
                // L1 - 6 bits tag [11:6] - cache size 32 KB, conflict for each 4 KB
                // L2 - 9 bits tag [14:6] - cache size 256 KB, conflict for each 32 KB
                // L3 - 13 bits tag [18:6] - cache size 8 MB, conflict for each 512 KB

                __m256 result256;
                __m256 a256_0, b256_0;    // AVX
                __m256 a256_1, b256_1;    // AVX
                __m256 a256_2;// , b256_2;    // AVX
                __m256 a256_3;// , b256_3;    // AVX
                __m256 c256_0, c256_1, c256_2, c256_3;
                __m256 c256_4, c256_5, c256_6, c256_7;

                c256_0 = _mm256_loadu_ps(&C[(0 + i)*ldc + (0 + j)]);
                c256_1 = _mm256_loadu_ps(&C[(1 + i)*ldc + (0 + j)]);
                c256_2 = _mm256_loadu_ps(&C[(0 + i)*ldc + (8 + j)]);
                c256_3 = _mm256_loadu_ps(&C[(1 + i)*ldc + (8 + j)]);

                c256_4 = _mm256_loadu_ps(&C[(2 + i)*ldc + (0 + j)]);
                c256_5 = _mm256_loadu_ps(&C[(3 + i)*ldc + (0 + j)]);
                c256_6 = _mm256_loadu_ps(&C[(2 + i)*ldc + (8 + j)]);
                c256_7 = _mm256_loadu_ps(&C[(3 + i)*ldc + (8 + j)]);

                // Pack up A
                float packedA[TILE_M * TILE_K];
                if(j == 0) {
                    for(int a = 0; a < TILE_M; a++) 
                    {
                        for(int b = 0; b < TILE_K; b++)
                            packedA[a * TILE_K + b] = A[(a + i) * lda + (b + k)];
                    }
                }
                // Pack up B
                // Packing up B results slower execution time
                /*float packedB[TILE_K * TILE_N];
                if(i == 0) {
                    for(int a = 0; a < TILE_K; a++)
                    {
                        for(int b = 0; b < TILE_N; b++)
                            packedB[a * TILE_N + b] = B[(a + k) * ldb + (b + j)];
                    }
                }*/

                for (k_d = 0; k_d < (TILE_K); ++k_d)
                {
                    /*a256_0 = _mm256_set1_ps(ALPHA*A[(0 + i)*lda + (k_d + k)]);
                    a256_1 = _mm256_set1_ps(ALPHA*A[(1 + i)*lda + (k_d + k)]);

                    a256_2 = _mm256_set1_ps(ALPHA*A[(2 + i)*lda + (k_d + k)]);
                    a256_3 = _mm256_set1_ps(ALPHA*A[(3 + i)*lda + (k_d + k)]);*/
                    a256_0 = _mm256_set1_ps(ALPHA * packedA[0 * TILE_K + k_d]);
                    a256_1 = _mm256_set1_ps(ALPHA * packedA[1 * TILE_K + k_d]);

                    a256_2 = _mm256_set1_ps(ALPHA * packedA[2 * TILE_K + k_d]);
                    a256_3 = _mm256_set1_ps(ALPHA * packedA[3 * TILE_K + k_d]);


                    b256_0 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (0 + j)]);
                    b256_1 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (8 + j)]);
                    /*b256_0 = _mm256_loadu_ps(&packedB[k_d * TILE_N + 0]);
                    b256_1 = _mm256_loadu_ps(&packedB[k_d * TILE_N + 8]);*/

                    // FMA - Intel Haswell (2013), AMD Piledriver (2012)
                    //c256_0 = _mm256_fmadd_ps(a256_0, b256_0, c256_0);
                    //c256_1 = _mm256_fmadd_ps(a256_1, b256_0, c256_1);
                    //c256_2 = _mm256_fmadd_ps(a256_0, b256_1, c256_2);
                    //c256_3 = _mm256_fmadd_ps(a256_1, b256_1, c256_3);

                    //c256_4 = _mm256_fmadd_ps(a256_2, b256_0, c256_4);
                    //c256_5 = _mm256_fmadd_ps(a256_3, b256_0, c256_5);
                    //c256_6 = _mm256_fmadd_ps(a256_2, b256_1, c256_6);
                    //c256_7 = _mm256_fmadd_ps(a256_3, b256_1, c256_7);

                    result256 = _mm256_mul_ps(a256_0, b256_0);
                    c256_0 = _mm256_add_ps(result256, c256_0);

                    result256 = _mm256_mul_ps(a256_1, b256_0);
                    c256_1 = _mm256_add_ps(result256, c256_1);

                    result256 = _mm256_mul_ps(a256_0, b256_1);
                    c256_2 = _mm256_add_ps(result256, c256_2);

                    result256 = _mm256_mul_ps(a256_1, b256_1);
                    c256_3 = _mm256_add_ps(result256, c256_3);


                    result256 = _mm256_mul_ps(a256_2, b256_0);
                    c256_4 = _mm256_add_ps(result256, c256_4);

                    result256 = _mm256_mul_ps(a256_3, b256_0);
                    c256_5 = _mm256_add_ps(result256, c256_5);

                    result256 = _mm256_mul_ps(a256_2, b256_1);
                    c256_6 = _mm256_add_ps(result256, c256_6);

                    result256 = _mm256_mul_ps(a256_3, b256_1);
                    c256_7 = _mm256_add_ps(result256, c256_7);
                }
                _mm256_storeu_ps(&C[(0 + i)*ldc + (0 + j)], c256_0);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (0 + j)], c256_1);
                _mm256_storeu_ps(&C[(0 + i)*ldc + (8 + j)], c256_2);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (8 + j)], c256_3);

                _mm256_storeu_ps(&C[(2 + i)*ldc + (0 + j)], c256_4);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (0 + j)], c256_5);
                _mm256_storeu_ps(&C[(2 + i)*ldc + (8 + j)], c256_6);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (8 + j)], c256_7);
            }

            for (j = (N / TILE_N)*TILE_N; j < N; ++j) {
                for (i_d = i; i_d < (i + TILE_M); ++i_d)
                {
                    for (k_d = k; k_d < (k + TILE_K); ++k_d)
                    {
                        PUT_IN_REGISTER float A_PART = ALPHA*A[i_d*lda + k_d];
                        C[i_d*ldc + j] += A_PART*B[k_d*ldb + j];
                    }
                }
            }
        }

        for (k = (K / TILE_K)*TILE_K; k < K; ++k)
        {
            for (i_d = i; i_d < (i + TILE_M); ++i_d)
            {
                PUT_IN_REGISTER float A_PART = ALPHA*A[i_d*lda + k];
                for (j = 0; j < N; ++j) {
                    C[i_d*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
    }

    for (i = (M / TILE_M)*TILE_M; i < M; ++i) {
        int j, k;
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }


    #pragma omp parallel for
    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] += bias[i];
        }
    }
}
}

#endif

void SRCNN::transpose(data_t *out, data_t *in, int in_row, int in_col)
{
    for(int i = 0; i < in_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[j * in_row + i] = in[i * in_col + j];
        }
    }
}

void SRCNN::activation(data_t *input, data_t *output, ImageDim inputDim, ACTIVATION activationType)
{
    switch(activationType)
    {
        case RELU:
            for(int k = 0; k < get<0>(inputDim); k++)
            {
                for(int i = 0; i < get<1>(inputDim); i++)
                {
                    for(int j = 0; j < get<2>(inputDim); j++)
                    {
                        output[(k * get<1>(inputDim) * get<2>(inputDim)) +
                               (i * get<1>(inputDim)) +
                                j] = 
                        relu_activate(input[(k * get<1>(inputDim) * get<2>(inputDim)) + (i * get<1>(inputDim)) + j]);
                    }
                }
            }
            break;
        default:
                cerr << "no such operation" << endl;
                exit(1);
            break;
    }
}

void SRCNN::readConvWeights(string filename, data_t *kernel, bool special/* = false*/, bool isReverse/* = false*/)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        exit(1);
    }

    int currentChannels = 1;
    if(special)
    {
        input >> currentChannels;
    }

    int kernelSizeSquare;
    input >> kernelSizeSquare;

    int nextChannels = 1;
    input >> nextChannels;

    if(isReverse)
    {
        for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
        {
            input >> kernel[i];
        }
    }
    else
    {
        for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
        {
            input >> kernel[i];
        }
    }

    input.close();
}

// read filter weight and change to NCHW format
void SRCNN::readConvWeights(string filename, data_t *kernel, KernelDim kernelDim, WeightFormat format, bool special)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        exit(1);
    }

    int currentChannels = 1;
    if(special)
    {
        input >> currentChannels;
    }

    int kernelSizeSquare;
    input >> kernelSizeSquare;

    int nextChannels = 1;
    input >> nextChannels;

    int totalSize = currentChannels * kernelSizeSquare * nextChannels;
    assert(totalSize == getTotalDimension(kernelDim));

    int num_filter = get<0>(kernelDim);
    int num_channel = get<1>(kernelDim);
    int num_height = get<2>(kernelDim);
    int num_width = get<3>(kernelDim);

    switch(format)
    {
        case NCHW:
            cout << "nchw" << endl;
            for(int n = 0; n < num_filter; n++)
            {
                for(int c = 0; c < num_channel; c++)
                {
                    for(int h = 0; h < num_height; h++)
                    {
                        for(int w = 0; w < num_width; w++)
                        {
                            input >> kernel[((n * num_channel + c) * num_height + h) * num_width + w];
                            /*cout << "index " << ((n * num_channel + c) * num_height + h) * num_width + w << " n " << n << " c " << c
                                << " h " << h << " w " << w << " " << kernel[((n * num_channel + c) * num_height + h) * num_width + w]
                                << endl;*/
                        }
                    }
                }
            }
            break;
        case NHWC:
            cout << "nhwc" << endl;
            for(int n = 0; n < num_filter; n++)
            {
                for(int h = 0; h < num_height; h++)
                {
                    for(int w = 0; w < num_width; w++)
                    {
                        for(int c = 0; c < num_channel; c++)
                        {
                            input >> kernel[((n * num_height + h) * num_width + w) * num_channel + c];
                        }
                    }
                }
            }
            break;
        case CHWN:
            cout << "chwn" << endl;
            for(int n = 0; n < num_filter; n++)
            {
                for(int h = 0; h < num_height; h++)
                {
                    for(int w = 0; w < num_width; w++)
                    {
                        for(int c = 0; c < num_channel; c++)
                        {
                            input >> kernel[((c * num_height + h) * num_width + w) * num_filter + n];
                            /*cout << "index " << ((c * num_height + h) * num_width + w) * num_filter + n 
                                << " c " << c
                                << " h " << h 
                                << " w " << w 
                                << " n " << n 
                                << " " << kernel[((c * num_height + h) * num_width + w) * num_filter + n]
                                << endl;*/
                        }
                    }
                }
            }
            break;
        case NCWH:
            cout << "ncwh" << endl;
            for(int n = 0; n < num_filter; n++)
            {
                for(int c = 0; c < num_channel; c++)
                {
                    for(int h = 0; h < num_height; h++)
                    {
                        for(int w = 0; w < num_width; w++)
                        {
                            input >> kernel[((n * num_channel + c) * num_width + w) * num_height + h];
                            /*cout << "index " << ((n * num_channel + c) * num_width + w) * num_height + h
                                << " n " << n
                                << " c " << c
                                << " w " << w
                                << " h " << h
                                << " " << kernel[((n * num_channel + c) * num_width + w) * num_height + h]
                                << endl;*/
                        }
                    }
                }
            }
            break;
        default:
            cerr << "no such format" << endl;
            exit(1);
            break;
    }

    input.close();
}

void SRCNN::readBiasWeights(string filename, data_t *kernel)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        exit(1);
    }

    int nextChannels;
    input >> nextChannels;

    int kernelSizeSquare;
    input >> kernelSizeSquare;

    for(int i = 0; i < nextChannels * kernelSizeSquare; i++)
    {
        input >> kernel[i];
    }

    input.close();
}

void SRCNN::testConvolution(data_t *input, data_t *output, ImageDim inputDim,
    ImageDim outputDim, data_t *kernels, KernelDim kernelDim, int stride/* = 1*/,
    data_t *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/,
    string outputConvWeightPath, string outputBiasWeightPath)
{
    int kernelOutputChannel = get<0>(kernelDim);
    int kernelInputChannel = get<1>(kernelDim);
    int kernelHeight = get<2>(kernelDim);
    int kernelWidth = get<3>(kernelDim);
    int kernelHeightSize = kernelHeight / 2;
    int kernelWidthSize = kernelWidth / 2;

    int inputChannel = get<0>(inputDim);
    int inputHeight = get<1>(inputDim);
    int inputWidth = get<2>(inputDim);

    int outputChannel = get<0>(outputDim);
    int outputHeight = get<1>(outputDim);
    int outputWidth = get<2>(outputDim);
    
    cout << outputConvWeightPath << endl;
    ofstream outputConvWeight(outputConvWeightPath);
    if(!outputConvWeight.is_open())
    {
        cout << "conv weight unsuccessful" << endl;
        exit(1);
    }
    cout << outputBiasWeightPath << endl;
    ofstream outputBiasWeight(outputBiasWeightPath);
    if(!outputBiasWeight.is_open())
    {
        cout << "bias weight unsuccessful" << endl;
        exit(1);
    }
    

    for(int k = 0; k < outputChannel; k++)
    {
        for(int n = 0; n < inputChannel; n++)
        {
#pragma omp parallel for
            for(int i = 0; i < inputHeight; i += stride)
            {
                for(int j = 0; j < inputWidth; j += stride)
                {
                    data_t sum = 0.0;
                    //output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] = 0;
                    for(int l = -kernelHeightSize; l <= kernelHeightSize; l++)
                    {
                        for(int m = -kernelWidthSize; m <= kernelWidthSize; m++)
                        {
                            int y = i + l;
                            int x = j + m;

                            // valid padding
                            x = x >= 0 ? (x < inputWidth ? x : inputWidth - stride) : 0;
                            y = y >= 0 ? (y < inputHeight ? y : inputHeight - stride) : 0;
                    
                            int inputIdx = (n * inputHeight * inputWidth) + (y * inputWidth) + x;

                            int kernelIdx = ((k * kernelInputChannel + n) 
                                            * kernelHeight + (l + kernelHeightSize))
                                            * kernelWidth + (m + kernelWidthSize);
                            sum += input[inputIdx] * kernels[kernelIdx]; 
                            
                            outputConvWeight << kernels[kernelIdx] << " ";
                            
                        }
                    }

                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += sum;
                }
            }
        }

        if(bias != NULL)
        {
#pragma omp parallel for
            for(int i = 0; i < outputHeight; i++)
            {
                for(int j = 0; j < outputWidth; j++)
                {
                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += bias[k];
                    outputBiasWeight << bias[(k * get<1>(biasDim) * get<2>(biasDim))] << " ";
                }
            }
        }
    }

    outputConvWeight.close();
    outputBiasWeight.close();    
}

void SRCNN::testReadConvWeights(string filename, string outputfile, data_t *kernel, bool special/* = false*/, bool isReverse/* = false*/)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        exit(1);
    }

    ofstream output(outputfile);
    if(!output.is_open())
    {
        cerr << "file " << outputfile << " opened unsuccessfully" << endl;
        exit(1);
    }

    int currentChannels = 1;
    if(special)
    {
        input >> currentChannels;
    }

    int kernelSizeSquare;
    input >> kernelSizeSquare;

    int nextChannels = 1;
    input >> nextChannels;

    if(isReverse)
    {
        for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
        {
            input >> kernel[i];
            output << kernel[i] << " ";
        }
    }
    else
    {
        for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
        {
            input >> kernel[i];
            output << kernel[i] << " ";
        }
    }

    input.close();
    output.close();
}

void SRCNN::testReadBiasWeights(string filename, string outputfile, data_t *kernel)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        exit(1);
    }

    ofstream output(outputfile);
    if(!output.is_open())
    {
        cerr << "file " << outputfile << " opened unsuccessfully" << endl;
        exit(1);
    }

    int nextChannels;
    input >> nextChannels;

    int kernelSizeSquare;
    input >> kernelSizeSquare;

    for(int i = 0; i < nextChannels * kernelSizeSquare; i++)
    {
        input >> kernel[i];
        output << kernel[i] << " ";
    }

    input.close();
    output.close();
}

void SRCNN::testWriteWeights(std::string outputfile, data_t *weights, ImageDim imageDim)
{
    ofstream out;
    out.open(outputfile);

    for(int i = 0; i < getTotalDimension(imageDim); i++)
    {
        out << weights[i] << endl;
    }

    out.close();
}

void SRCNN::testWriteWeights(std::string outputfile, data_t *weights, KernelDim kernelDim)
{
    ofstream out;
    out.open(outputfile);

    for(int i = 0; i < getTotalDimension(kernelDim); i++)
    {
        out << weights[i] << endl;
    }

    out.close();
}
