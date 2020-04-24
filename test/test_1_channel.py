import numpy as np
from scipy import ndimage

Input = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

kernel = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0]
])

bias = -1

stride = 1
padding = 1

# In deep learning, the term 'convolution' means element-wise multiplication
# then sum each element,
# which is as same as 'correlation' in mathematic.
output = ndimage.correlate(Input, kernel, mode='constant', cval=0.0)

print(output)
