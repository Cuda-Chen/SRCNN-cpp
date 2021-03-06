import numpy as np

Input = np.array([
    [[1, 2, 0, 0, 1],
     [0, 0, 1, 2, 1],
     [0, 0, 1, 1, 1],
     [0, 1, 0, 1, 0],
     [1, 2, 1, 1, 1]],

    [[1, 1, 2, 1, 2],
     [2, 0, 1, 1, 1],
     [2, 0, 2, 2, 0],
     [2, 2, 2, 1, 2],
     [2, 1, 2, 0, 2]],

    [[1, 2, 1, 1, 2],
     [2, 2, 2, 1, 1],
     [1, 0, 1, 0, 2],
     [2, 1, 1, 1, 1],
     [1, 2, 2, 0, 2]]
])

filters = np.array([
    [
    [[0, -1, 0],
     [-1, 1, 0],
     [0, -1, -1]],

    [[-1, 1, 0],
     [-1, -1, 1],
     [1, -1, 1]],

    [[0, 0, 1],
     [1, 0, 0],
     [-1, 1, 1]]
    ],

    [
    [[1, -1, 1],
     [0, -1, 1],
     [1, 0, 1]],

    [[0, -1, -1],
     [1, 1, 1],
     [0, -1, 0]],

    [[-1, 0, -1],
     [-1, -1, -1],
     [0, 0, 1]]
    ]
])

biases = np.array([
    [[1]],
    [[2]]
])

print(Input.shape)
print(filters.shape)
print(biases.shape)
