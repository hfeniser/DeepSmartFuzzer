import sys
sys.path.append('../')

import numpy as np
from utils import get_layer_outs_optimized, calc_major_func_regions, percent_str
from math import floor

import time

def measure_k_multisection_cov(model, test_inputs, k, train_inputs=None, major_func_regions=None, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs_optimized(model, test_inputs, skip=skip)

    if major_func_regions is None:
        if train_inputs is None:
            raise ValueError("Training inputs must be provided when major function regions are not given")

        major_func_regions = calc_major_func_regions(model, train_inputs, skip)

    activation_table_by_section, upper_activation_table, lower_activation_table = {}, {}, {}
    neuron_set = set()

    t = time.time()

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        for out_for_input in layer_out:  # out_for_input is output of layer for single input
            for neuron_index in range(out_for_input.shape[-1]):
                neuron_out = np.mean(out_for_input[..., neuron_index])
                global_neuron_index = (layer_index, neuron_index)

                neuron_set.add(global_neuron_index)

                neuron_low = major_func_regions[layer_index][0][neuron_index]
                neuron_high = major_func_regions[layer_index][1][neuron_index]
                section_length = (neuron_high - neuron_low) / k
                section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0
                section_index = section_index if section_index < k else k-1

                activation_table_by_section[(global_neuron_index, section_index)] = True

                if neuron_out < neuron_low:
                    lower_activation_table[global_neuron_index] = True
                elif neuron_out > neuron_high:
                    upper_activation_table[global_neuron_index] = True

    multisection_activated = len(activation_table_by_section.keys())
    lower_activated = len(lower_activation_table.keys())
    upper_activated = len(upper_activation_table.keys())

    total = len(neuron_set)

    n2i = {n:i for i,n in enumerate(neuron_set)}
    
    multisection_coverage_state = np.zeros((total, k))
    for n,section_index in activation_table_by_section:
        ni = n2i[n]
        multisection_coverage_state[ni, section_index] = 1
    
    multisection_coverage_state = multisection_coverage_state.reshape(-1)
    
    lower_coverage_state = np.zeros(total)
    for n in lower_activation_table:
        i = n2i[n]
        lower_coverage_state[i] = 1
    
    upper_coverage_state = np.zeros(total)
    for n in upper_activation_table:
        i = n2i[n]
        upper_coverage_state[i] = 1
        
    upperlower_coverage_state = np.concatenate((upper_coverage_state, lower_coverage_state), axis=0)
    

    return (percent_str(multisection_activated, k*total), multisection_activated,
            percent_str(upper_activated+lower_activated, 2 * total),
            percent_str(upper_activated, total),
            lower_activated, upper_activated, total,
            multisection_activated, upper_activated, lower_activated, total, outs,
            multisection_coverage_state, upperlower_coverage_state, upper_coverage_state)