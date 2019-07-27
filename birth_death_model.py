#from model import Model
import numpy as np
from numba import jitclass
from numba import char, float64, int64

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
]

@jitclass(spec)
class BirthDeathModel():
    def __init__(self):
        self.S = np.array([[-1, 1], [1, -1]], dtype=np.int64)
        self.rate_constants = np.array([1, 1], dtype=np.float64)
          
    def get_intensities(self, x):
        temp = np.array([x[0], x[1]], dtype=np.float64)
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)
    