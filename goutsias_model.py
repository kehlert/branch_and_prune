import numpy as np
from numba import jitclass
from numba import char, float64, int64

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
]

@jitclass(spec)
class GoutsiasModel():
    def __init__(self):
        self.S = np.array([[1, -1,  0,  0,  0,  0,  0,  0, -2,  2],
                           [0,  0,  0,  0, -1,  1, -1,  1,  1, -1],
                           [0,  0,  1, -1,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0, -1,  1,  0,  0,  0,  0],
                           [0,  0,  0,  0,  1, -1, -1,  1,  0,  0],
                           [0,  0,  0,  0,  0,  0,  1, -1,  0,  0]])
        AV = 6.0221415 * 10**8
        self.rate_constants = np.array([0.043, 0.0007, 0.0715, 0.0039, 0.012E9 / AV,
                                        0.4791, 0.00012E9 / AV, 0.8765E-11, 0.05E9 / AV, 0.5])

    def get_intensities(self, x):
        temp = np.array([x[2], x[0], x[4], x[2], x[1] * x[3],
                         x[4], x[1] * x[4], x[5], 0.5 * x[0] * (x[0]-1), x[1]], dtype=np.float64)
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)