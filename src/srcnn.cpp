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

    resize(this->gray, this->bicubic, Size(), this->scale, this->scale, INTER_CUBIC); // using bicubic to resize input image

    // read conv and bias weights
    Dim conv1WeightsDim = make_tuple(64, 9, 9);
    Dim conv2WeightsDim = make_tuple(32, 5, 5);
    Dim conv3WeightsDim = make_tuple(1, 5, 5);
    double conv1Weights = ;
    double conv2Weights = ;
    double conv3Weights = ;
    double bias1Weights = ;
    double bias2Weights = ;
    double bias3Weights = ;

    // conv1
    // conv2
    // conv3
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
    KernelDim kernelDim = make_tuple(1, 1, kernelHeight, kernelWidth);

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
                        
                            sum += input[(n * inputHeight * inputWidth) + (y * inputWidth) + x] * kernels[(k * kernelHeight * kernelWidth) + ((l + kernelHeight) * kernelWidth) + (m + kernelWidth)];
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
        cerr << "file " << weightPath << " opened unsuccessfully" << endl;
        return;
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

void SRCNN:readBiasWeights(string filename, double *kernel)
{
    ifstream input(filename);
    if(!input.is_open())
    {
        cerr << "file " << filename << " opened unsuccessfully" << endl;
        return;
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

