#include <string>
#include <tuple>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "srcnn.hpp"

#include "gaussian.hpp"

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
    resize(this->gray, this->downsample, Size(), this->scale / 2, this->scale / 2, INTER_CUBIC);

    // using bicubic to resize input image
    resize(this->downsample, this->bicubic, Size(), this->scale, this->scale, INTER_CUBIC);

    // prepare input, output, and conv data
    int inputWidth = this->bicubic.cols;
    int inputHeight = this->bicubic.rows;
    ImageDim inputDim = make_tuple(1, inputHeight, inputWidth);
    double *input = new double[inputHeight* inputWidth];
    ImageDim conv1Dim = make_tuple(64, inputHeight, inputWidth);
    ImageDim conv2Dim = make_tuple(32, inputHeight, inputWidth);
    ImageDim conv3Dim = make_tuple(1, inputHeight, inputWidth);
    double *conv1Data = new double[getTotalDimension(conv1Dim)];
    double *conv2Data = new double[getTotalDimension(conv2Dim)];
    double *conv3Data = new double[getTotalDimension(conv3Dim)];
    int outputWidth = inputWidth;
    int outputHeight = outputHeight;
    double *dst = new double[outputHeight * outputWidth];
    for(int i = 0; i < inputHeight; i++)
    {
        for(int j = 0; j < inputWidth; j++)
        {
            input[(i * inputWidth) + j] = this->bicubic.at<uchar>(i, j) / 255.0;
            dst[(i * inputWidth) + j] = 0;
        }
    }

    // read conv and bias weights
    KernelDim conv1WeightsDim = make_tuple(1, 9, 9, 64);
    KernelDim conv2WeightsDim = make_tuple(64, 5, 5, 32);
    KernelDim conv3WeightsDim = make_tuple(32, 5, 5, 1);
    ImageDim bias1Dim = make_tuple(64, 1, 1);
    ImageDim bias2Dim = make_tuple(32, 1, 1);
    ImageDim bias3Dim = make_tuple(1, 1, 1);
    double *conv1Weights = new double[getTotalDimension(conv1WeightsDim)];
    double *conv2Weights = new double[getTotalDimension(conv2WeightsDim)];
    double *conv3Weights = new double[getTotalDimension(conv3WeightsDim)];
    double *bias1Weights = new double[getTotalDimension(bias1Dim)];
    double *bias2Weights = new double[getTotalDimension(bias2Dim)];
    double *bias3Weights = new double[getTotalDimension(bias3Dim)];
    readConvWeights(this->weightsConv1, conv1Weights);
    readConvWeights(this->weightsConv2, conv2Weights, true);
    readConvWeights(this->weightsConv3, conv3Weights, false, true);
    readBiasWeights(this->biasConv1, bias1Weights);
    readBiasWeights(this->biasConv2, bias2Weights);
    readBiasWeights(this->biasConv3, bias3Weights);

    // conv1 (feature extraction)
    convolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim);
    activation(conv1Data, conv1Data, conv1Dim, RELU);
    // conv2 (non-linear mapping)
    convolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights, conv2WeightsDim, 1, bias2Weights, bias2Dim);
    activation(conv2Data, conv2Data, conv2Dim, RELU);
    // conv3 (reconstruction)
    convolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim);

    for(int i = 0; i < outputHeight; i++)
    {
        for(int j = 0; j < outputWidth; j++)
        {
            dst[(i * outputWidth) + j] = conv3Data[((1 - 1) * get<1>(conv3Dim) + i) * get<2>(conv3Dim) + j];
        }
    }

    // copy to output OpenCV Mat
    Mat SRCNN(outputHeight, outputWidth, CV_64FC1, dst);
    this->output = SRCNN;
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
        vector<double> contents;
        double temp;
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

void SRCNN::testConv(string filename)
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
    double *input = new double[1 * height * width];
    double *output = new double[1 * height * width];
    ImageDim inputDim = make_tuple(1, height, width);
    ImageDim outputDim = make_tuple(1, height, width);
    unsigned char *dst = new unsigned char[height * width];

    int kernelWidth = 9;
    int kernelHeight = 9;
    double sigma = 3.0;

    // Conv test
    double *kernel = new double[kernelHeight * kernelWidth];
    KernelDim kernelDim = make_tuple(1, kernelHeight, kernelWidth, 1);

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

    gaussianFilter(source, destination, width, height, kernelWidth, kernelHeight, sigma);
    Mat result(height, width, CV_8UC1, destination);
    imshow("gaussian", result);
    waitKey(0);

    convolution(input, output, inputDim, outputDim, kernel, kernelDim); 
    int counter = 0;
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

// standard convolution
void SRCNN::convolution(double *input, double *output, ImageDim inputDim,
    ImageDim outputDim, double *kernels, KernelDim kernelDim, int stride/* = 1*/,
    double *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/)
{
    int kernelInputChannel = get<0>(kernelDim);
    int kernelHeight = get<1>(kernelDim);
    int kernelWidth = get<2>(kernelDim);
    int kernelOutputChannel = get<3>(kernelDim);
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
            for(int i = 0; i < inputHeight; i += stride)
            {
                for(int j = 0; j < inputWidth; j += stride)
                {
                    double sum = 0.0;
                    for(int l = -kernelHeightSize; l <= kernelHeightSize; l++)
                    {
                        for(int m = -kernelWidthSize; m <= kernelWidthSize; m++)
                        {
                            int y = i + l;
                            int x = j + m;

                            // zero padding
                            x = x >= 0 ? (x < inputWidth ? x : inputWidth - stride) : 0;
                            y = y >= 0 ? (y < inputHeight ? y : inputHeight - stride) : 0;
                        
                            /*sum += input[(n * inputHeight * inputWidth) + (y * inputWidth) + x] * 
                                kernels[(n * kernelHeight * kernelWidth) + 
                                        ((l + kernelHeight) * kernelWidth) + 
                                        (m + kernelWidth)];*/
                                sum += input[(n * inputHeight * inputWidth) + (y * inputWidth) + x] * 
                                    kernels[(((n) * kernelHeight + 
                                            (l + kernelHeight)) * kernelWidth + 
                                            (m + kernelWidth)) * kernelOutputChannel + 
                                            k]; // sum += input[n][y][x] * kernels[n][l + kernelHeight][m + kernelWidth][k]
                        }
                    }

                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] = sum;
                    if(bias != NULL)
                    {
                        output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += bias[(k * get<1>(biasDim) * get<2>(biasDim))];
                    } 
                }
            }
        }

        if(bias != NULL)
        {
            for(int i = 0; i < outputHeight; i++)
            {
                for(int j = 0; j < outputWidth; j++)
                {
                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += bias[(k * get<1>(biasDim) * get<2>(biasDim))];
                }
            }
        }
    }
}

void SRCNN::activation(double *input, double *output, ImageDim inputDim, ACTIVATION activationType)
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

void SRCNN::readConvWeights(string filename, double *kernel, bool special/* = false*/, bool isReverse/* = false*/)
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
        cin >> currentChannels;
    }

    int kernelSizeSquare;
    cin >> kernelSizeSquare;

    int nextChannels = 1;
    cin >> nextChannels;

    if(isReverse)
    {
        for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
        {
            cin >> kernel[i];
        }
    }
    else
    {
        for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
        {
            cin >> kernel[i];
        }
    }

    input.close();
}

void SRCNN::readBiasWeights(string filename, double *kernel)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        exit(1);
    }

    int nextChannels;
    cin >> nextChannels;

    int kernelSizeSquare;
    cin >> kernelSizeSquare;

    for(int i = 0; i < nextChannels * kernelSizeSquare; i++)
    {
        cin >> kernel[i];
    }

    input.close();
}

