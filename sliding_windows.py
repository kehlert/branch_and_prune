import numpy as np
import itertools

def get_support(model_data, alpha):
    #TODO deal with conservation laws
    model = model_data.model
    x0 = model_data.x0
    R = model.S.shape[1]
    
    support = {tuple(x0)}
    bounds = [[0,0] for r in range(0, R)]
    intensities = model_data.t * model.get_intensities(x0)
    
    scaled_std_dev = alpha * np.sqrt(intensities)
    upper_bounds = np.ceil(intensities + scaled_std_dev).astype(int) #upper bound
    lower_bounds = np.maximum(np.floor(intensities - scaled_std_dev), 0).astype(int) #lower bound
    bounds = [range(lb, ub+1) for lb,ub in zip(lower_bounds, upper_bounds)]
    print(bounds)
    all_reaction_combos = np.asarray(list(itertools.product(*bounds)))
    print(all_reaction_combos.dot(model.S.T))
   
        
    return support