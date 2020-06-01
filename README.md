# SRCNN-cpp
C++ implementation of SRCNN (Super-Resolution Convolutional Neural Network)

# Note
Currently I am facing the issue ringing effect of output. Any help will be 
delighted.

The weight format is **CHWN** because the weights is created by cuda-convnet.

# How to Build and Run
We provide two ways to build this project: CMake and QMake

If you just want to build the program and quickly test it, type these:
```
$ mkdir build && cd build && cmake .. && make && cd ..
$ SRCNN_cpp <input image path>
```

If you want much faster speed with OpenMP, follow these steps:
1. Set `i_want_openmp` from `OFF` to `ON` in `CMakeLists.txt`.
2. Follow the usual CMake building steps then run it.

If you would like to use QMake just run the `SRCNN-cpp.pro`. Also, the OpenMP in QMake is ON by default.

# Note
This implementation uses 9-5-5 architecture.
- 9-5-5 means f1 = 9, f2 = 5, f3 = 5.
- You can see the explanation in [Section 4.3.2 in SRCNN paper](https://arxiv.org/pdf/1501.00092.pdf).

# TODO
- [x] `im2col`
- [x] validate convolution using [this image](https://zhuanlan.zhihu.com/p/63974249)
- [x] padding feature

# Contributors
- [Cuda-Chen](https://github.com/Cuda-Chen) Cuda Chen - creator, maintainer
- [masc4ii](https://github.com/masc4ii) masc4ii - OpenMP and QMake utility
