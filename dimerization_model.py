import numpy as np

class DimerizationModel():
    def __init__(self):
        self.S = np.array([[ 1,  0, 0],
                           [ 0,  1, 0],
                           [ 0, -2, 1],
                           [-1,  0, 0],
                           [ 0, -1, 0]]).transpose()
              
        self.rate_constants = np.array([25, 100, 0.001, 0.1, 1])
        
    def get_intensities(self, x):
        temp = np.array([1, x[0], x[1] * (x[1] - 1), x[0], x[1]])
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)