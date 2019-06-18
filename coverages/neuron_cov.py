import sys
sys.path.append('../')

import numpy as np
from utils import get_layer_outs, percent_str
from collections import defaultdict


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def measure_neuron_cov(model, test_inputs, scaler=scale, threshold=0, skip_layers=None, outs=None):
    if outs is None:
        outs = get_layer_outs(model, test_inputs, skip_layers)

    activation_table = defaultdict(bool)

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        for out_for_input in layer_out[0]:  # out_for_input is output of layer for single input
            out_for_input = scaler(out_for_input)

            for neuron_index in range(out_for_input.shape[-1]):
                activation_table[(layer_index, neuron_index)] = activation_table[(layer_index, neuron_index)] or\
                                                                np.mean(out_for_input[..., neuron_index]) > threshold
    covered = len([1 for c in activation_table.values() if c])
    total = len(activation_table.keys())
    
    coverage_state = np.zeros(total)
    for i,c in enumerate(activation_table.values()):
        if c:
            coverage_state[i] = 1
        else:
            coverage_state[i] = 0

    return percent_str(covered, total), covered, total, outs, coverage_state 