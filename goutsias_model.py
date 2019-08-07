import numpy as np
from numba import jitclass
from numba import char, float64, int64

#https://www.researchgate.net/profile/Markus_Hegland/publication/37629076_A_Krylov-based_Finite_State_Projection_algorithm_for_solving_the_chemical_master_equation_arising_in_the_discrete_modelling_of_biological_systems/links/0deec5239c8d72ed8f000000/A-Krylov-based-Finite-State-Projection-algorithm-for-solving-the-chemical-master-equation-arising-in-the-discrete-modelling-of-biological-systems.pdf

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
]

@jitclass(spec)
class GoutsiasModel():
    def __init__(self):
        #[M, DNA, RNA, D, DNA.D, DNA.2D]
        
        self.S = np.array([[1, -1,  0,  0,  0,  0,  0,  0, -2,  2],
                           [0,  0,  0,  0, -1,  1,  0,  0,  1,  0],
                           [0,  0,  1, -1,  0,  0,  0,  0,  0,  0],
                           [0,  0,  0,  0, -1,  1, -1,  1,  0, -1],
                           [0,  0,  0,  0,  1, -1, -1,  1,  0,  0],
                           [0,  0,  0,  0,  0,  0,  1, -1,  0,  0]])
        AV = 6.0221415 * 10**8
        self.rate_constants = np.array([0.043, 0.0007, 0.078, 0.0039, 0.012E9 / AV,
                                        0.4791, 0.00012E9 / AV, 0.8765E-11, 0.05E9 / AV, 0.5])

    def get_intensities(self, x):
        temp = np.array([x[2], x[0], x[4], x[2], x[1] * x[3],
                         x[4], x[3] * x[4], x[5], 0.5 * x[0] * (x[0]-1), x[3] * (x[3]-1)], dtype=np.float64)
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)