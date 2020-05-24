#include <iostream>
#include <iomanip>
#include <string>
#include <tuple>
#include <fstream>
#include <cassert>

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
    float *input = new float[inputHeight * inputWidth];
    ImageDim conv1Dim = make_tuple(64, inputHeight, inputWidth);
    ImageDim conv2Dim = make_tuple(32, inputHeight, inputWidth);
    ImageDim conv3Dim = make_tuple(1, inputHeight, inputWidth);
    float *conv1Data = new float[getTotalDimension(conv1Dim)];
    float *conv2Data = new float[getTotalDimension(conv2Dim)];
    float *conv3Data = new float[getTotalDimension(conv3Dim)];
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;
    cout << "output width height " << outputWidth << " " << outputHeight << endl;
    float *dst = new float[outputHeight * outputWidth];
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
    KernelDim conv1WeightsDim = make_tuple(64, 1, 9, 9);
    KernelDim conv2WeightsDim = make_tuple(32, 64, 5, 5);
    KernelDim conv3WeightsDim = make_tuple(1, 32, 5, 5);
    cout << "biasDim" << endl;
    ImageDim bias1Dim = make_tuple(64, 1, 1);
    ImageDim bias2Dim = make_tuple(32, 1, 1);
    ImageDim bias3Dim = make_tuple(1, 1, 1);
    cout << "finish setting bias dim" << endl;
    float *conv1Weights = new float[getTotalDimension(conv1WeightsDim)];
    float *conv1Weights_transposed = new float[getTotalDimension(conv1WeightsDim)];
    float *conv2Weights = new float[getTotalDimension(conv2WeightsDim)];
    float *conv2Weights_transposed = new float[getTotalDimension(conv2WeightsDim)];
    float *conv3Weights = new float[getTotalDimension(conv3WeightsDim)];
    float *conv3Weights_transposed = new float[getTotalDimension(conv3WeightsDim)];
    float *bias1Weights = new float[getTotalDimension(bias1Dim)];
    float *bias2Weights = new float[getTotalDimension(bias2Dim)];
    float *bias3Weights = new float[getTotalDimension(bias3Dim)]; 
    cout << "finish allocating conv and bias weights' space" << endl;
    
    readConvWeights(this->weights[0], conv1Weights); cout << "weight[0]" << endl;
    readConvWeights(this->weights[1], conv2Weights, true); cout << "weight[1]" << endl;
    readConvWeights(this->weights[2], conv3Weights, false, true); cout << "weight[2]" << endl;
    readBiasWeights(this->weights[3], bias1Weights); cout << "weight[3]" << endl;
    readBiasWeights(this->weights[4], bias2Weights); cout << "weight[4]" << endl;
    readBiasWeights(this->weights[5], bias3Weights); cout << "weight[5]" << endl;
    
    /*
    testReadConvWeights(this->weights[0], "myConv1Weight.txt", conv1Weights); cout << "weight[0]" << endl;
    testReadConvWeights(this->weights[1], "myConv2Weight.txt", conv2Weights, true); cout << "weight[1]" << endl;
    testReadConvWeights(this->weights[2], "myConv3Weight.txt", conv3Weights, false, true); cout << "weight[2]" << endl;
    testReadBiasWeights(this->weights[3], "myBias1Weight.txt", bias1Weights); cout << "weight[3]" << endl;
    testReadBiasWeights(this->weights[4], "myBias2Weight.txt", bias2Weights); cout << "weight[4]" << endl;
    testReadBiasWeights(this->weights[5], "myBias3Weight.txt", bias3Weights); cout << "weight[5]" << endl;
    */

    // conv1 (feature extraction)
    cout << "conv1" << endl;
    transpose(conv1Weights_transposed, conv1Weights, getTotalDimension(conv1WeightsDim) / get<0>(conv1WeightsDim), get<0>(conv1WeightsDim));
    convolution(input, conv1Data, inputDim, conv1Dim, conv1Weights_transposed, conv1WeightsDim, 1, bias1Weights, bias1Dim);
    //convolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim);
    /*testConvolution(input, conv1Data, inputDim, conv1Dim, conv1Weights, conv1WeightsDim, 1, bias1Weights, bias1Dim,
        "myConv1Weight.txt", "myBias1Weight.txt");*/
    activation(conv1Data, conv1Data, conv1Dim, RELU);
//#if 0 
    float *conv1arr = new float[get<1>(conv1Dim) * get<2>(conv1Dim)];
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
//#endif

    // conv2 (non-linear mapping)
    cout << "conv2" << endl;
    transpose(conv2Weights_transposed, conv2Weights, 64 * 5 * 5, 32);// CHWN -> NCHW
    convolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights_transposed, conv2WeightsDim, 1, bias2Weights, bias2Dim);
    //convolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights, conv2WeightsDim, 1, bias2Weights, bias2Dim);
    /*testConvolution(conv1Data, conv2Data, conv1Dim, conv2Dim, conv2Weights, conv2WeightsDim, 1, bias2Weights, bias2Dim, 
        "myConv2Weight.txt", "myBias2Weight.txt");*/
    activation(conv2Data, conv2Data, conv2Dim, RELU);
//#if 0
    float *conv2arr = new float[get<1>(conv2Dim) * get<2>(conv2Dim)];
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
//#endif

    // conv3 (reconstruction)
    cout << "conv3" << endl;
    transpose(conv3Weights_transposed, conv3Weights, getTotalDimension(conv3WeightsDim) / get<0>(conv3WeightsDim), get<0>(conv3WeightsDim));
    convolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights_transposed, conv3WeightsDim, 1, bias3Weights, bias3Dim);
    //convolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim);
    /*testConvolution(conv2Data, conv3Data, conv2Dim, conv3Dim, conv3Weights, conv3WeightsDim, 1, bias3Weights, bias3Dim,
        "myConv3Weight.txt", "myBias3Weight.txt");*/
    //activation(conv3Data, conv3Data, conv3Dim, RELU);
//#if 0
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
            }
        }
        Mat conv3(get<1>(conv3Dim), get<2>(conv3Dim), CV_8UC1, conv3arr);
        //conv3.convertTo(conv3, CV_8UC1, 255.0);
        string outputname = "conv3_" + to_string(i) + ".jpg";
        imwrite(outputname, conv3);
    }
    delete [] conv3arr;
//#endif

    cout << "prepare output" << endl;
    for(int i = 0; i < outputHeight; i++)
    {
        for(int j = 0; j < outputWidth; j++)
        {
            //cout << i << " " << j << " fine" << endl;
            dst[(i * outputWidth) + j] = conv3Data[((1 - 1) * get<1>(conv3Dim) + i) * get<2>(conv3Dim) + j];
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
    Mat SRCNN(outputHeight, outputWidth, CV_32FC1, dst);
    //Mat SRCNN(outputHeight, outputWidth, CV_64FC1, conv3Data);
    this->output = SRCNN;

    // dump weights
    testWriteWeights("myWeightConv1Dump", conv1Weights_transposed, conv1WeightsDim);
    testWriteWeights("myWeightConv2Dump", conv2Weights_transposed, conv2WeightsDim);
    testWriteWeights("myWeightConv3Dump", conv3Weights_transposed, conv3WeightsDim);
    testWriteWeights("myWeightBias1Dump", bias1Weights, bias1Dim);
    testWriteWeights("myWeightBias2Dump", bias2Weights, bias2Dim);
    testWriteWeights("myWeightBias3Dump", bias3Weights, bias3Dim);

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
        vector<float> contents;
        float temp;
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
    float *input = new float[1 * height * width];
    float *output = new float[1 * height * width];
    ImageDim inputDim = make_tuple(1, height, width);
    ImageDim outputDim = make_tuple(1, height, width);
    unsigned char *dst = new unsigned char[height * width];

    int kernelWidth = 3;
    int kernelHeight = 3;
    float sigma = 3.0;

    // Conv test
    double *kernel = new double[kernelHeight * kernelWidth];
    float *kernel_float = new float[kernelHeight * kernelWidth];
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
    for(int i = 0; i < kernelHeight; i++)
    {
        for(int j = 0; j < kernelWidth; j++)
        {
            kernel_float[i * kernelWidth + j] = (float)kernel[i * kernelWidth + j];
        }
    }

    gaussianFilter(source, destination, width, height, kernelWidth, kernelHeight, sigma);
    Mat result(height, width, CV_8UC1, destination);
    imshow("gaussian", result);
    waitKey(0);

    convolution(input, output, inputDim, outputDim, kernel_float, kernelDim); 
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

void SRCNN::testConv1Channel()
{
    // input
    ImageDim inputDim = make_tuple(1, 5, 5);
    float input[]
    {
     1, 2, 3, 4, 5,
     6, 7, 8, 9, 10,
     11, 12, 13, 14, 15,
     16, 17, 18, 19, 20,
     21, 22, 23, 24, 25
    };

    // kernel
    KernelDim kernelDim = make_tuple(1, 1, 3, 3);
    float kernel[]
    {
     0, 0, 0,
     0, 1, 0,
     0, 0, 0
    };

    // output
    ImageDim outputDim = make_tuple(1, 5, 5);
    float *output = new float[getTotalDimension(outputDim)];

    // bias
    ImageDim biasDim = make_tuple(1, 1, 1);
    float bias[] = { 0 };

    // apply convolution
    convolution(input, output, inputDim, outputDim,
                kernel, kernelDim, 1, bias, 
                biasDim);

    // print the convoluted result
    int outputHeight = get<1>(outputDim);
    int outputWidth = get<2>(outputDim);
    for(int i = 0; i < get<0>(outputDim); i++)
    {
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
    float input[] = 
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
    float *output = new float[getTotalDimension(outputDim)];
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
    KernelDim filtersDim = make_tuple(2, 3, 3, 3);
    float filters[] = 
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
    float biases[] = 
    {/* b0 */
     1,
     /* b1 */
     0
    };

    // operate convolution on test data
    convolution(input, output, inputDim,
                outputDim, filters, filtersDim, 2,
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

// matrix transpose test
void SRCNN::testTranspose()
{
    // kernel
    KernelDim filtersDim = make_tuple(2, 3, 3, 3);
    int kernel_num = 2;
    int kernel_c = 3;
    int kernel_h = 3;
    int kernel_w = 3;
    float testKernel[] = 
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

    float *filters_transposed = new float[getTotalDimension(filtersDim)];
    int filters_transposed_h = kernel_c * kernel_h * kernel_w;
    int filters_transposed_w = kernel_num;

    float *result = new float[getTotalDimension(filtersDim)];
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
    float *conv1Weights = new float[getTotalDimension(conv1WeightsDim)];
    float *conv1Weights_transposed = new float[getTotalDimension(conv1WeightsDim)];
    float *conv1Weights_tt = new float[getTotalDimension(conv1WeightsDim)];
    float *conv2Weights = new float[getTotalDimension(conv2WeightsDim)];
    float *conv2Weights_transposed = new float[getTotalDimension(conv2WeightsDim)];
    float *conv2Weights_tt = new float[getTotalDimension(conv2WeightsDim)];
    float *conv3Weights = new float[getTotalDimension(conv3WeightsDim)];
    float *conv3Weights_transposed = new float[getTotalDimension(conv3WeightsDim)];
    float *conv3Weights_tt = new float[getTotalDimension(conv3WeightsDim)];
    
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
    KernelDim kernelDim = make_tuple(32, 64, 5, 5);
    float *weight = new float[getTotalDimension(kernelDim)];
    readConvWeights(this->weights[1], weight, kernelDim, CHWN, true);
}

// standard convolution
void SRCNN::convolution(float *input, float *output, ImageDim inputDim,
    ImageDim outputDim, float *kernels, KernelDim kernelDim, int stride/* = 1*/,
    float *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/)
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
    int input_col_height = kernelHeight * kernelWidth * inputChannel;
    int input_col_width = outputHeight * outputWidth;
    float *input_col = new float[input_col_height * input_col_width];

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

    delete [] input_col;
}


/* From Berkeley Vision's Caffe!
 * https://github.com/BVLC/caffe/blob/master/LICENSE
 */
void SRCNN::im2col(float *data_im, ImageDim imageDim, KernelDim kernelDim,
                   int stride, int pad, float *data_col)
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

void SRCNN::col2im(float *data_col, ImageDim imageDim, KernelDim kernelDim,
                int stride, int pad, float *data_im)
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
        for(int h = 0; h < col_height; h++)
        {
            for(int w = 0; w < col_width; w++)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_idx = (c * col_height + h) * col_width + w;
                float value = data_col[col_idx];
                col2imAddPixel(data_im, imageDim, im_row, im_col,
                               c_im, pad, value);
            }
        }
    }
}

float SRCNN::im2colGetPixel(float *im, ImageDim imageDim, 
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

void SRCNN::col2imAddPixel(float *im, ImageDim imageDim,
                           int row, int col, int channel, int pad, float value)
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

void SRCNN::matMul(float *out, float *kernel, float *in, float *bias,
                   int kernel_row, int kernel_col, int in_row, int in_col)
{
    if(bias == NULL)
    {
        naiveGEMM(out, kernel, in, 
                  kernel_row, kernel_col, in_row, in_col);
    }
    else
    {
        naiveGEMM_addBias(out, kernel, in, bias,
                          kernel_row, kernel_col, in_row, in_col);
    }
}
 
void SRCNN::naiveGEMM(float *out, float *kernel, float *in,
                      int kernel_row, int kernel_col, int in_row, int in_col)
{
     /* The output matrix dimension will be kernel_row * in_col */

    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] = 0;
            for(int k = 0; k < in_row; k++)
            {
                out[i * in_col + j] +=
                    kernel[i * kernel_col + k] *
                    in[k * in_col + j];
            }
        }
    }
}

void SRCNN::naiveGEMM_addBias(float *out, float *kernel, float *in, float *bias,
                              int kernel_row, int kernel_col, int in_row, int in_col)
{
    /* The output matrix dimension will be kernel_row * in_col */

    for(int i = 0; i < kernel_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[i * in_col + j] = 0;
            for(int k = 0; k < in_row; k++)
            {
                out[i * in_col + j] +=
                    kernel[i * kernel_col + k] *
                    in[k * in_col + j];
            }
            out[i * in_col + j] += bias[i];
        }
    }
}

void SRCNN::transpose(float *out, float *in, int in_row, int in_col)
{
    for(int i = 0; i < in_row; i++)
    {
        for(int j = 0; j < in_col; j++)
        {
            out[j * in_row + i] = in[i * in_col + j];
        }
    }
}

void SRCNN::activation(float *input, float *output, ImageDim inputDim, ACTIVATION activationType)
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

void SRCNN::readConvWeights(string filename, float *kernel, bool special/* = false*/, bool isReverse/* = false*/)
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

void SRCNN::readConvWeights(string filename, float *kernel, KernelDim kernelDim, WeightFormat format, bool special)
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

    switch(format)
    {
        case NCHW:
            cout << "nchw" << endl;
            break;
        case NHWC:
            cout << "nhwc" << endl;
            break;
        case CHWN:
            cout << "chwn" << endl;
            break;
        default:
            cerr << "no such format" << endl;
            exit(1);
            break;
    }
 
    for(int i = 0; i < currentChannels * kernelSizeSquare * nextChannels; i++)
    {
        input >> kernel[i];
    }

    input.close();
}

void SRCNN::readBiasWeights(string filename, float *kernel)
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

void SRCNN::testConvolution(float *input, float *output, ImageDim inputDim,
    ImageDim outputDim, float *kernels, KernelDim kernelDim, int stride/* = 1*/,
    float *bias/* = NULL*/, ImageDim biasDim/* = make_tuple(0, 0, 0)*/,
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
                    float sum = 0.0;
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

void SRCNN::testReadConvWeights(string filename, string outputfile, float *kernel, bool special/* = false*/, bool isReverse/* = false*/)
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

void SRCNN::testReadBiasWeights(string filename, string outputfile, float *kernel)
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

void SRCNN::testWriteWeights(std::string outputfile, float *weights, ImageDim imageDim)
{
    ofstream out;
    out.open(outputfile);

    for(int i = 0; i < getTotalDimension(imageDim); i++)
    {
        out << weights[i] << endl;
    }

    out.close();
}

void SRCNN::testWriteWeights(std::string outputfile, float *weights, KernelDim kernelDim)
{
    ofstream out;
    out.open(outputfile);

    for(int i = 0; i < getTotalDimension(kernelDim); i++)
    {
        out << weights[i] << endl;
    }

    out.close();
}
