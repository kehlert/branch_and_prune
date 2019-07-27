def get_support(model_data, init_dist, r):
    model = model_data.model
    R = model.S.shape[1]
    
    
    support = {tuple(x0) for x0 in init_dist}
    diff = support
    
    for i in range(0, r):
        new_states = set()
        for state in diff:
            reachable_states = state + model.S.T
            for j in range(0, R):
                if any(reachable_states[j,:] < 0):
                    continue
                new_states.add(tuple(reachable_states[j,:]))
                     
        diff = new_states - support
        support = support.union(new_states)
        
    return support