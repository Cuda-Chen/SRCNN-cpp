#ifndef GAUSSIAN_HPP
#define GAUSSIAN_HPP

#include "datatype.hpp"

void generateKernel(int width, int height, data_t sigma, data_t *kernel);
void gaussianFilter(unsigned char *src, unsigned char *dst,
	int width, int height, int kernelWidth, int kernelHeight, data_t sigma);

#endif
