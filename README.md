# SRCNN-cpp
C++ implementation of SRCNN (Super-Resolution Convolutional Neural Network)

# How to Build and Run
```
$ mkdir build && cd build && cmake .. && make && cd ..
$ SRCNN_cpp <input image path>
```

# Note
This implementation uses 9-5-5 architecture.
- 9-5-5 means f1 = 9, f2 = 5, f3 = 5.
- You can see the explanation in [Section 4.3.2 in SRCNN paper](https://arxiv.org/pdf/1501.00092.pdf).
