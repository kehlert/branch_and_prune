import numpy as np
from numba import jitclass
from numba import char, float64, int64

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
]

@jitclass(spec)
class GORDEModel():
    def __init__(self):
        self.S = np.array([[-1,  1],
                           [1,  -1]]).transpose()
        self.rate_constants = np.array([0.5, 0.7])

    def get_intensities(self, x):
        temp = np.array([x[0] > 0, x[1] > 0], dtype=np.float64)
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)