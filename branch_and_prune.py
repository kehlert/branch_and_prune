import numpy as np

from scipy.stats import poisson
from scipy.stats import multinomial
from scipy.sparse import csc_matrix

import queue

from dataclasses import dataclass, field
from typing import Any

import r_step_reachability

@dataclass()       
class ReactionComboBranch:
    combo: tuple
    parent_state_id: int
    intensity: float
    initial_state_id: int
    log_multinomial_pmf: float

@dataclass(order=True)
class QueueItem:
    priority: float
    branch: Any=field(compare=False)
        
    def __init__(self, neg_priority, branch):
        self.priority = -neg_priority #lowest priority items are retrieved first
        self.branch = branch
        
#initial_dist is a dict with keys=state tuples, vals=probabilities
#states not in initial_dist are assumed to have probability 0
def get_support(model_data, initial_dist, eps = 10**(-6)):
    model = model_data.model
    dt = model_data.t
    R = model.S.shape[1]
    
    #keys are states, and the values are the states' assigned unique id's
    support = dict()

    #these arrays are for constructing the (sparse) generator matrix
    entries = []
    rows = [] #'from' states
    cols = [] #'to' states
    sink_id = 0 #state id of the sink state
    
    branch_queue = queue.PriorityQueue()
    
    expanded_initial_dist = initial_dist.copy()
    
    for state, prob in initial_dist.items():
        x0 = np.array(state)
        intensities = model.get_intensities(x0)
        
        #'smooth out' the initial distribution if any intensities are zero
        if np.any(intensities == 0):
            #expand by 1 reaction
            smoothed_support = r_step_reachability.get_support(model_data, {state: prob}, 1)
            for additional_state in smoothed_support:
                if additional_state not in initial_dist:
                    expanded_initial_dist[additional_state] = prob
  
    initial_dist_list = list(expanded_initial_dist.items())
    initial_states = np.zeros((len(expanded_initial_dist), len(model_data.species)))
    initial_prob = np.zeros(len(expanded_initial_dist))
    initial_dist_intensities = np.zeros((len(initial_dist_list), R))
                    
    for i, (state, prob) in enumerate(initial_dist_list):
        initial_states[i,:] = np.array(state)
        initial_prob[i] = prob

        x0 = initial_states[i,:]
        euler_intensities = model.get_intensities(x0)
        #midpoint_intensities = model.get_intensities(x0 + model.S.dot(dt / 2.0 * euler_intensities))
        #initial_dist_intensities[index,:] = midpoint_intensities
        initial_dist_intensities[i,:] = euler_intensities

        #lmbda0 = np.sum(midpoint_intensities)
        lmbda0 = np.sum(euler_intensities)

        R = model.S.shape[1]
        branch = ReactionComboBranch((np.zeros(R)), -1, -1, i, 0.)
        branch_queue.put(QueueItem(np.log(prob), branch)) #the minus sign is so highest are removed first   
    
    while not branch_queue.empty():
        #get the most likely combo
        queue_item = branch_queue.get()
        branch = queue_item.branch
        
        if branch.parent_state_id >= 0:
            #parent_state_id < 0 if the branch is a root
            entries.append(branch.intensity)
            rows.append(branch.parent_state_id)
        
        #after applying the reactions encoded in combo, state_array is the resulting state
        state_array = initial_states[branch.initial_state_id,:] + np.dot(model.S, branch.combo)
        
        if any(state_array < 0):
            cols.append(sink_id)
            continue #prune

        state = tuple(state_array.astype(int))
        intensities = model.get_intensities(state_array)
        
        if state in support:
            #we visited the state before
            if branch.parent_state_id >= 0:
                #parent_state_id < 0 if the branch is a root
                cols.append(support[state])
            continue #prune
        else:
            #this is a new state, so add it to the support, and then we should branch
            state_index = len(support) + 1 #add 1, because 0 is the sink index
            support[state] = state_index
            
            if branch.parent_state_id >= 0:
                #parent_state_id < 0 if the branch is a root
                cols.append(state_index)
            
            #add the diagonal entry
            entries.append(-np.sum(intensities))
            rows.append(state_index)
            cols.append(state_index)
        
        #below this line is the code to add new branches to the queue
        new_combos = branch.combo + np.eye(R)

        for r in range(0, R):
            new_combo = new_combos[r,:]
            
            log_prob_bound, log_multinomial_pmf = get_log_prob_bound(new_combo,
                                                                     initial_dist_intensities[branch.initial_state_id,:],
                                                                     dt,
                                                                     branch.log_multinomial_pmf,
                                                                     r,
                                                                     initial_prob[branch.initial_state_id])
    
            if log_prob_bound >= np.log(eps) and intensities[r] > 0:
                #add branch to queue
                new_branch = ReactionComboBranch(new_combo,
                                                 state_index,
                                                 intensities[r],
                                                 branch.initial_state_id,
                                                 log_multinomial_pmf)
                branch_queue.put(QueueItem(log_prob_bound, new_branch))
            else:
                #prune
                entries.append(intensities[r])
                rows.append(state_index)
                cols.append(sink_id)

    #add 1 to the shape to account for the sink
    generator = csc_matrix((entries, (rows, cols)), shape=(len(support)+1, len(support)+1))
    
    #the zeroth element is the sink
    initial_dist_array = np.zeros(len(support)+1)
    for state, prob in initial_dist.items():
        initial_dist_array[support[state]] = prob
    initial_dist_array[0] = 1. - np.sum(initial_dist_array)
    
    return support, generator, initial_dist_array
    
def get_log_prob_bound(y, intensities, dt, parent_log_multinomial_pmf, reaction_index, initial_state_prob):
    lambda0 = np.sum(intensities)
    y_sum = np.sum(y)
    
    if intensities[reaction_index] == 0:
        return -np.Inf, 0
    
    log_multinomial_pmf = parent_log_multinomial_pmf\
                          + np.log(intensities[reaction_index] / lambda0)\
                          - np.log(y[reaction_index])\
                          + np.log(y_sum)

    lambda0_dt = lambda0 * dt
    
    if y_sum > lambda0_dt:   
        log_chernoff_bound = y_sum - lambda0_dt - np.log(y_sum / lambda0_dt) * y_sum
    else:
        log_chernoff_bound = 0.

    log_prob_bound = np.log(initial_state_prob) + log_multinomial_pmf + log_chernoff_bound  

    return log_prob_bound, log_multinomial_pmf