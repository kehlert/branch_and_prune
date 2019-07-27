#from Dinh and Sidje 2016

#from model import Model
import numpy as np
from numba import jitclass
from numba import char, float64, int64

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
]

@jitclass(spec)
class GeneToggleModel():
    def __init__(self):
        self.S = np.array([[ 1,  0],
                           [-1,  0],
                           [ 0,  1],
                           [ 0, -1]], dtype=np.int64).transpose()
        self.rate_constants = np.array([50., 1., 50., 1.], dtype=np.float64)
          
    def get_intensities(self, x):
        temp = np.array([1. / (1 + 2*x[1]),
                         x[0],
                         1. / (1 + 2*x[0]),
                         x[1]], dtype=np.float64)
        temp = np.multiply(temp, self.rate_constants)
        return np.maximum(temp, 0)
        
    