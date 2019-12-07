#include <cmath>

#include "gaussian.hpp"

const double PI = 3.14159;

void generateKernel(int width, int height, double sigma, double *kernel)
{
	double sum = 0.0;
	int strideWidth = width / 2;
	int strideHeight = height / 2;
	
	// generate kernel
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] = exp(-(pow(i - strideHeight, 2) + pow(j - strideWidth, 2)) / (2 * sigma * sigma))
				/ (2 * PI * sigma * sigma);
			sum += kernel[(i * width) + j];
		}
	}

	// then normalize each element
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] /= sum;
		}
	}
}

void gaussianFilter(unsigned char *src, unsigned char *dst,
	int width, int height, int kernelWidth, int kernelHeight, double sigma)
{
	double *kernel = new double[kernelWidth * kernelHeight];

	generateKernel(kernelWidth, kernelHeight, sigma, kernel);

	int strideWidth = kernelWidth / 2;
	int strideHeight = kernelHeight / 2;

	for(int row = 0 + strideHeight; row < height - strideHeight; row++)
	{
		for(int col = 0 + strideWidth; col < width - strideWidth; col++)
		{
			double temp = 0.0;
			int xindex;
			int yindex;
			
			for(int krow = 0; krow < kernelHeight; krow++)
			{
				for(int kcol = 0; kcol < kernelWidth; kcol++)
				{
					xindex = krow + row - strideHeight;
					yindex = kcol + col - strideWidth;
					temp += src[(xindex * width) + yindex] * kernel[(krow * kernelWidth) + kcol];
				}
			}

			if(temp > 255)
			{
				temp = 255;
			}
			else if(temp < 0)
			{
				temp = 0;
			}

			dst[(row * width) + col] = (unsigned char)temp;
		}
	} 

	delete [] kernel;
}
