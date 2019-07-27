from scipy.sparse import csc_matrix
import numpy as np

#support should be a set of tuples, where the tuples correspond to different states (e.g. (100, 200) for a two-species model)
def get_generator(model_data, init_dist, support):
    #the zeroth state is the sink
    state_to_index = {state: i+1 for i, state in enumerate(support)}
    index_to_state = {index: state for state,index in state_to_index.items()}
    rows = [0] #state we end up in
    cols = [0] #state we started in
    entries = [0]
    
    model = model_data.model
    R = model.S.shape[1]
    
    for state in support:
        reachable_states = state + model.S.T
        state_index = state_to_index[state]
        intensities = model.get_intensities(np.array(state))
        
        #add diagonal entry
        rows.append(state_index)
        cols.append(state_index)
        entries.append(-np.sum(intensities))
        
        for r in range(0, R):
            cols.append(state_index)
            reachable_state = tuple(reachable_states[r,:])
            entries.append(intensities[r])
            
            if reachable_state in support:
                rows.append(state_to_index[reachable_state])
            else:
                rows.append(0) #sink state
                
    generator = csc_matrix((entries, (rows, cols)), shape=(len(support)+1, len(support)+1))
    
    #the zeroth element is the sink
    initial_dist = np.zeros(len(support)+1)
    for state, prob in init_dist.items():
        initial_dist[state_to_index[state]] = prob
    
    return generator, initial_dist, index_to_state