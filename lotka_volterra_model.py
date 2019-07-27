import numpy as np
from numba import jitclass
from numba import char, float64, int64

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
]

@jitclass(spec)
class LotkaVolterraModel():
    def __init__(self):
        self.S = np.array([[ 1,  0],
                           [-1,  1],
                           [ 0, -1]]).transpose()
        self.rate_constants = np.array([2, 0.01, 2])

    def get_intensities(self, x):
        temp = np.array([x[0], x[0] * x[1], x[1]], dtype=np.float64)
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)