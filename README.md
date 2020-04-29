# SRCNN-cpp
C++ implementation of SRCNN (Super-Resolution Convolutional Neural Network)

# Note
Currently I am facing the issue ringing effect of output. Any help will be 
delighted.

# How to Build and Run
```
$ mkdir build && cd build && cmake .. && make && cd ..
$ SRCNN_cpp <input image path>
```

# Note
This implementation uses 9-5-5 architecture.
- 9-5-5 means f1 = 9, f2 = 5, f3 = 5.
- You can see the explanation in [Section 4.3.2 in SRCNN paper](https://arxiv.org/pdf/1501.00092.pdf).

# TODO
- [x] `im2col`
- [x] validate convolution using [this image](https://zhuanlan.zhihu.com/p/63974249)
- [x] padding feature
