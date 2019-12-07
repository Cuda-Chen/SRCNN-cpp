#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

void generateKernel(int width, int height, double sigma, double *kernel);
void gaussianFilter(unsigned char *src, unsigned char *dst,
	int width, int height, int kernelWidth, int kernelHeight, double sigma);

#endif
