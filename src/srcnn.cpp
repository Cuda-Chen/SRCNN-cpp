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
    double *input = new double[inputHeight * inputWidth];
    ImageDim conv1Dim = make_tuple(64, inputHeight, inputWidth);
    ImageDim conv2Dim = make_tuple(32, inputHeight, inputWidth);
    ImageDim conv3Dim = make_tuple(1, inputHeight, inputWidth);
    double *conv1Data = new double[getTotalDimension(conv1Dim)];
    double *conv2Data = new double[getTotalDimension(conv2Dim)];
    double *conv3Data = new double[getTotalDimension(conv3Dim)];
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;
    cout << "output width height " << outputWidth << " " << outputHeight << endl;
    double *dst = new double[outputHeight * outputWidth];
    cout << "assign input and output value" << endl;
    for(int i = 0; i < inputHeight; i++)
    {
        for(int j = 0; j < inputWidth; j++)
        {
            input[(i * inputWidth) + j] = this->bicubic.at<uchar>(i, j) / 255.0;
            dst[(i * inputWidth) + j] = 0;
        }
    }

    // read conv and bias weights
    cout << "read conv and bias weights" << endl;
    cout << "kernelDim" << endl;
    KernelDim conv1WeightsDim = make_tuple(1, 9, 9, 64);
    KernelDim conv2WeightsDim = make_tuple(64, 5, 5, 32);
    KernelDim conv3WeightsDim = make_tuple(32, 5, 5, 1);
    cout << "biasDim" << endl;
    ImageDim bias1Dim = make_tuple(64, 1, 1);
    ImageDim bias2Dim = make_tuple(32, 1, 1);
    ImageDim bias3Dim = make_tuple(1, 1, 1);
    cout << "finish setting bias dim" << endl;
    double *conv1Weights = new double[getTotalDimension(conv1WeightsDim)];
    double *conv2Weights = new double[getTotalDimension(conv2WeightsDim)];
    double *conv3Weights = new double[getTotalDimension(conv3WeightsDim)];
    double *bias1Weights = new double[getTotalDimension(bias1Dim)];
    double *bias2Weights = new double[getTotalDimension(bias2Dim)];
    double *bias3Weights = new double[getTotalDimension(bias3Dim)]; 
    cout << "finish allocating conv and bias weights' space" << endl;
    /*
    readConvWeights(this->weights[0], conv1Weights); cout << "weight[0]" << endl;
    readConvWeights(this->weights[1], conv2Weights, true); cout << "weight[1]" << endl;
    readConvWeights(this->weights[2], conv3Weights, false, true); cout << "weight[2]" << endl;
    readBiasWeights(this->weights[3], bias1Weights); cout << "weight[3]" << endl;
    readBiasWeights(this->weights[4], bias2Weights); cout << "weight[4]" << endl;
    readBiasWeights(this->weights[5], bias3Weights); cout << "weight[5]" << endl;
    */
    testReadConvWeights(this->weights[0], "myConv1Weight.txt", conv1Weights); cout << "weight[0]" << endl;
    testReadConvWeights(this->weights[1], "myConv2Weight.txt", conv2Weights, true); cout << "weight[1]" << endl;
    testReadConvWeights(this->weights[2], "myConv3Weight.txt", conv3Weights, false, true); cout << "weight[2]" << endl;
    testReadBiasWeights(this->weights[3], "myBias1Weight.txt", bias1Weights); cout << "weight[3]" << endl;
    testReadBiasWeights(this->weights[4], "myBias2Weight.txt", bias2Weights); cout << "weight[4]" << endl;
    testReadBiasWeights(this->weights[5], "myBias3Weight.txt", bias3Weights); cout << "weight[5]" << endl;


    // conv1 (feature extraction)
    cout << "conv1" << endl;
    convolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim);
    /*testConvolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim,
        "myConv1Weight.txt", "myBias1Weight.txt");*/
    activation(conv1Data, conv1Data, conv1Dim, RELU); 
    /* 
    for(int i = 0; i < 64; i++)
    {
        Mat conv1(get<1>(conv1Dim), get<2>(conv1Dim), CV_64FC1, conv1Data[i * get<1>(conv1Dim) * get<2>(conv1Dim)]);
        conv1.convertTo(conv1, CV_8UC1, 255.0);
        string outputname = "conv1_" + to_string(i) + ".jpg";
        imwrite(outputname, conv1);
    }
    */

    // conv2 (non-linear mapping)
    cout << "conv2" << endl;
    convolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights, conv2WeightsDim, 1, bias2Weights, bias2Dim);
    /*testConvolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights, conv2WeightsDim, 1, bias2Weights, bias2Dim, 
        "myConv2Weight.txt", "myBias2Weight.txt");*/
    activation(conv2Data, conv2Data, conv2Dim, RELU);
    /*
    for(int i = 0; i < 32; i++)
    {
        Mat conv2(get<1>(conv2Dim), get<2>(conv2Dim), CV_64FC1, conv2Data[i * get<1>(conv2Dim) * get<2>(conv2Dim)]);
        conv2.convertTo(conv2, CV_8UC1, 255.0);
        string outputname = "conv2_" + to_string(i) + ".jpg";
        imwrite(outputname, conv2);
    }
    */

    // conv3 (reconstruction)
    cout << "conv3" << endl;
    convolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim);
    /*testConvolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim,
        "myConv3Weight.txt", "myBias3Weight.txt");*/
    cout << "prepare output" << endl;
    for(int i = 0; i < outputHeight; i++)
    {
        for(int j = 0; j < outputWidth; j++)
        {
            //cout << i << " " << j << " fine" << endl;
            dst[(i * outputWidth) + j] = conv3Data[((1 - 1) * get<1>(conv3Dim) + i) * get<2>(conv3Dim) + j];
            //dst[(i * outputWidth) + j] = conv3Data[(i * outputWidth) + j];
            if(dst[(i * outputWidth) + j] != 0)
            {
                cout << "index " << i << " " << j << " " << conv3Data[(i * outputWidth) + j] << endl;
            }
        }
    }

    // copy to output OpenCV Mat
    cout << "copy to output OpenCV Mat" << endl;
    Mat SRCNN(outputHeight, outputWidth, CV_64FC1, dst);
    //Mat SRCNN(outputHeight, outputWidth, CV_64FC1, conv3Data);
    this->output = SRCNN;

    delete [] input;
    delete [] conv1Data;
    delete [] conv2Data;
    delete [] conv3Data;
    delete [] conv1Weights;
    delete [] conv2Weights;
    delete [] conv3Weights;
    delete [] bias1Weights;
    delete [] bias2Weights;
    delete [] bias3Weights;
    //delete [] dst; 
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

// http://cs231n.github.io/convolutional-networks/
void SRCNN::testConv()
{
    // input
    ImageDim inputDim = make_tuple(3, 5, 5);
    double input[] = 
    {/* channel 0 */
     2, 1, 1, 1, 2,
     1, 2, 0, 2, 2,
     1, 0, 0, 1, 0,
     1, 0, 2, 0, 1,
     0, 1, 2, 2, 1,
     /* channel 1 */
     1, 2, 1, 2, 1,
     2, 2, 0, 2, 0,
     1, 0, 2, 0, 1,
     0, 0, 2, 0, 2,
     1, 2, 0, 2, 0,
     /* channel 2*/
     0, 0, 0, 0, 1,
     2, 1, 2, 2, 1,
     1, 0, 1, 1, 2,
     1, 0, 2, 1, 0,
     1, 0, 0, 1, 0
    };

    // output
    int outputDepth = 2;
    int outputHeight = 3;
    int outputWidth = 3;
    ImageDim outputDim = make_tuple(2, 3, 3);
    double *output = new double[getTotalDimension(outputDim)];
    for(int i = 0; i < outputDepth; i++)
    {
        for(int j = 0; j < outputHeight; j++)
        {
            for(int k = 0; k < outputWidth; k++)
            {
                output[(i * outputHeight + j) * outputWidth + k] = 0;
            }
        }
    }

    // kernel
    KernelDim filtersDim = make_tuple(3, 3, 3, 2);
    double filters[] = 
    {/* filter w0 */
     /* channel 0 */
     1, 1, -1,
     1, -1, -1,
     -1, 0, 0,
     /* channel 1 */
     1, 0, -1,
     1, 1, 0,
     1, 1, -1,
     /* channel 2 */
     1, 1, 1,
     0, -1, 1,
     1, 0, 0,

     /* filter w1 */
     /* channel 0 */
     0, -1, 0,
     0, 1, 0,
     1, -1, 1,
     /* channel 1 */
      1, 0, -1,
     -1, -1, 1,
     0, -1, 0,
     /* channel 2 */
     0, 0, -1,
     -1, -1, -1,
     -1, 0, 0
    };

    // bias
    ImageDim biasesDim = make_tuple(2, 1, 1);
    double biases[] = 
    {/* b0 */
     1,
     /* b1 */
     0
    };

    // operate convolution on test data
    convolution(input, output, inputDim,
                outputDim, filters, filtersDim, 1,
                biases, biasesDim);

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

                                // sum += input[n][y][x] * kernels[n][l + kernelHeight][m + kernelWidth][k]
                                int inputIdx = (n * inputHeight * inputWidth) + (y * inputWidth) + x;
                                int kernelIdx = (((n) * kernelHeight + 
                                            (l + kernelHeightSize)) * kernelWidth + 
                                            (m + kernelWidthSize)) * kernelOutputChannel + 
                                            k;
                                sum += input[inputIdx] * kernels[kernelIdx]; 
                        }
                    }

                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] = sum;
                    /*
                    if(bias != NULL)
                    {
                        output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += bias[(k * get<1>(biasDim) * get<2>(biasDim))];
                    }
                    */ 
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

void SRCNN::readBiasWeights(string filename, double *kernel)
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

void SRCNN::testConvolution(double *input, double *output, ImageDim inputDim,
    ImageDim outputDim, double *kernels, KernelDim kernelDim, int stride/* = 1*/,
    double *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/,
    string outputConvWeightPath, string outputBiasWeightPath)
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

    /*
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
    */

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
                                int inputIdx = (n * inputHeight * inputWidth) + (y * inputWidth) + x;
                                int kernelIdx = (((n) * kernelHeight + 
                                            (l + kernelHeight)) * kernelWidth + 
                                            (m + kernelWidth)) * kernelOutputChannel + 
                                            k;
                                sum += input[inputIdx] * kernels[kernelIdx]; 
                                
                                //outputConvWeight << kernels[kernelIdx] << " ";
                                
                        }
                    }

                    output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] = sum;
                    /*
                    if(bias != NULL)
                    {
                        output[(k * outputHeight * outputWidth) + (i * outputWidth) + j] += bias[(k * get<1>(biasDim) * get<2>(biasDim))];
                    }
                    */ 
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
                    //outputBiasWeight << bias[(k * get<1>(biasDim) * get<2>(biasDim))] << " ";
                }
            }
        }
    }

    /*
    outputConvWeight.close();
    outputBiasWeight.close();
    */
}

void SRCNN::testReadConvWeights(string filename, string outputfile, double *kernel, bool special/* = false*/, bool isReverse/* = false*/)
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

void SRCNN::testReadBiasWeights(string filename, string outputfile, double *kernel)
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

