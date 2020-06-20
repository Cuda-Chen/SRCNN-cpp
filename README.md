# SRCNN-cpp
Pure C++ implementation of SRCNN (Super-Resolution Convolutional Neural Network)

# How to Build and Run
We provide two ways to build this project: CMake and QMake

## CMake
If you just want to build the program and quickly test it, type these:
```
$ mkdir build && cd build && cmake .. && make && cd ..
$ SRCNN_cpp <input image path>
```

If you want much faster speed with OpenMP, follow these steps:
1. Set `i_want_openmp` from `OFF` to `ON` in `CMakeLists.txt`.
2. Follow the usual CMake building steps then run it.

## QMake
Just run the `SRCNN-cpp.pro`. Also, the OpenMP option in QMake is set to ON by default.

# Note
This implementation uses 9-5-5 architecture.
- 9-5-5 means f1 = 9, f2 = 5, f3 = 5.
- You can see the explanation in [Section 4.3.2 in SRCNN paper](https://arxiv.org/pdf/1501.00092.pdf).

# TODO
- [x] `im2col`
- [x] validate convolution using [this image](https://zhuanlan.zhihu.com/p/63974249)
- [x] padding feature

# Note
- The weight format is **CHWN**. For more information, download the author's
training code [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).
    - The weight format conversion and storing code is in `saveFilters.m`.

# Contributors
- [Cuda-Chen](https://github.com/Cuda-Chen) Cuda Chen - creator, maintainer
- [masc4ii](https://github.com/masc4ii) masc4ii - OpenMP and QMake utility
