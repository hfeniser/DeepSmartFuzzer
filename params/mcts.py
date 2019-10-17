import numpy as np
import itertools

from params.parameters import Parameters

mcts = Parameters()

def tc1(state): 
    # limit the level/depth of root
    return state.level > 8

mcts.tc1 = tc1

def tc2(iterations):
    # limit the number of iterations on root
    return iterations > 25

mcts.tc2 = tc2

def tc3(state):
    original_input = state.original_input
    mutated_input = state.mutated_input

    alpha, beta = 0.1, 0.5
    if(np.sum((original_input-mutated_input) != 0) < alpha * np.sum(original_input>0)):
        return not np.max(np.abs(mutated_input-original_input)) <= 255
    else:
        return not np.max(np.abs(mutated_input-original_input)) <= beta*255

mcts.tc3 = tc3

mcts.action_division_p1 = (1,3,3,1)

translation = list(itertools.product(["translation"], [(-5,-5), (-5,0), (0,-5), (0,0), (5,0), (0,5), (5,5)]))
rotation = list(itertools.product(["rotation"], [-15,-12,-9,-6,-3,3,6,9,12,15]))
contrast = list(itertools.product(["contrast"], [1.2+0.2*k for k in range(10)]))
brightness = list(itertools.product(["brightness"], [10+10*k for k in range(10)]))
blur = list(itertools.product(["blur"], [k+1 for k in range(10)]))

mcts.actions_p2 = contrast + brightness + blur

def calc_implicit_reward_neuron(p1, p2):
    distance = np.abs(p1-p2)
    implicit_reward =  1 / (distance + 1)
    #print("p1, p2, distance, implicit_reward:", p1, p2, distance, implicit_reward)
    return implicit_reward

mcts.calc_implicit_reward_neuron = calc_implicit_reward_neuron

def calc_implicit_reward(activation_values, covered_positions):
    #print("activation_values, covered_positions", activation_values, covered_positions)
    return np.max(activation_values * np.logical_not(covered_positions))

mcts.calc_implicit_reward = calc_implicit_reward
mcts.save_batch = False