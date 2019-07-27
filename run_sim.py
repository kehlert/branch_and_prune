import numpy as np
from numba import jit

@jit(nopython=True)
def run_sim(model, x0, t_end):
    R = model.rate_constants.size
    buffer_size = 10**4

    x = x0
    t = 0
    
    #first column is the time
    state_path = np.empty((buffer_size, len(x0) + 1))
    reactions_path = []
    state_path_index = 0

    while True:
        intensities = model.get_intensities(x)
        total_intensity = np.sum(intensities)

        state_path[state_path_index,0] = t
        state_path[state_path_index,1:] = x
        state_path_index += 1
        
        if state_path_index >= buffer_size:
            state_path = np.vstack((state_path, np.empty((buffer_size, len(x0) + 1))))
            
        #get time to next reaction
        dt = np.random.exponential(1 / total_intensity)

        #check if simulation is over
        if t + dt > t_end:
            t = t_end
            state_path[state_path_index,0] = t
            state_path[state_path_index,1:] = x
            state_path_index += 1
            break

        t += dt

        #get and execute the next reaction
        probabilities = intensities / total_intensity

        #due to numpy rounding errors,
        #it often thinks the probabilities sum to more than 1
        temp = np.random.multinomial(1, probabilities * 0.9999999999999)
        reaction_index = np.argmax(temp)
        reactions_path.append(reaction_index)
        x = x + model.S[:,reaction_index]

    return state_path[0:state_path_index, :], reactions_path