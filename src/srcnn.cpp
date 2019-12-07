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
    Dim inputDim = make_tuple(1, height, width);
    Dim outputDim = make_tuple(1, height, width);
    unsigned char *dst = new unsigned char[height * width];

    int kernelWidth = 9;
    int kernelHeight = 9;
    double sigma = 3.0;

    // Conv test
    double *kernel = new double[kernelHeight * kernelWidth];
    Dim kernelDim = make_tuple(1, kernelHeight, kernelWidth);

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
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            dst[(i * width) + j] = output[(i * width) + j] * 255.0;
        }
    }
    Mat result1(height, width, CV_8UC1, dst);
    imshow("gaussian 1", result1);
    waitKey(0);
}

void SRCNN::convolution(double *input, double *output, Dim inputDim,
    Dim outputDim, double *kernels, Dim kernelDim, int stride/* = 1*/,
    double *bias/* = NULL*/, Dim biasDim/* = make_tuple(0, 0, 0)*/)
{
    int kernelHeight = get<1>(kernelDim);
    int kernelWidth = get<2>(kernelDim);
    int kernelHeightSize = kernelHeight / 2;
    int kernelWidthSize = kernelWidth / 2;
    int channel = get<0>(outputDim);
    int height = get<1>(outputDim);
    int width = get<2>(outputDim);
    for(int k = 0; k < channel; k++)
    {
        for(int i = 0; i < height; i += stride)
        {
            for(int j = 0; j < width; j += stride)
            {
                double sum = 0.0;
                for(int l = -kernelHeightSize; l <= kernelHeightSize; l++)
                {
                    for(int m = -kernelWidthSize; m <= kernelWidthSize; m++)
                    {
                        int y = i + l;
                        int x = j + m;

                        // zero padding
                        x = x >= 0 ? (x < width ? x : width - stride) : 0;
                        y = y >= 0 ? (y < height ? y : height - stride) : 0;
                    
                        sum += input[(k * height * width) + (y * width) + x] * kernels[(k * kernelHeight * kernelWidth) + ((l + kernelHeight) * kernelWidth) + (m + kernelWidth)];
                    }
                }

                if(bias != NULL)
                {
                    output[(k * height * width) + (i * width) + j] = sum + bias[(k * get<1>(biasDim) * get<2>(biasDim))];
                }
            }
        }
    }
}

void SRCNN::activation(double *input, double *output, Dim inputDim, ACTIVATION activationType)
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

