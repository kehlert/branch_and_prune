import numpy as np
import copy

def get_support(model_data, eps):
    model = model_data.model
    dt = model_data.t
    x0 = model_data.x0
    R = model.S.shape[1]
    
    tau = eps
    
    grad = {tuple(x0): 1.}
    support = set()
    
    m = 0

    while True:
        m += 1
        
        if tau < 10**(-50):
            break
            
        tilde_states = set()
        
        for state, u in grad.items(): 
            reachable_states = state + model.S.T
            #tilde_states.add(state)
            
            for i in range(0, R):
                reachable_state = reachable_states[i, :]
                if any(reachable_state < 0):
                    continue
                
                tilde_states.add(tuple(reachable_state))

        grad_tilde = {}
        
        for state in tilde_states:
            prev_states = state - model.S.T
            u = 0.
             
            for i in range(0, R):
                prev_state = prev_states[i,:]
                intensities = model.get_intensities(prev_state)
                total_intensity = np.sum(intensities)  

                key = tuple(prev_state)
                if key in grad:          
                    u += intensities[i] / total_intensity * (1. - np.exp(-total_intensity * dt)) * grad[key]
                                                         
            grad_tilde[state] = u

        sorted_states = sorted(grad_tilde.items(), key=lambda kv: kv[1])
        u_sum = 0.0
        index = 0
        
        while True:
            if index == len(sorted_states):
                break
                
            state_u_pair = sorted_states[index]
            state = state_u_pair[0]
            u = state_u_pair[1]

            u_sum += u
            
            index += 1
            if u_sum >= tau:
                u_sum -= u
                index -= 1
                break

        if index == len(sorted_states):
            #we are done
            support = support.union(set(grad_tilde.keys()))
            break
                                         
        tau = tau - u_sum
        grad = dict(sorted_states[index:])
        new_states = set(grad.keys())
        
        support = support.union(new_states)

    return support