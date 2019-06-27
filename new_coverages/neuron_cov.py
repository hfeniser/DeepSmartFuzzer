import sys
sys.path.append('../')

import numpy as np
from utils import get_layer_outs_new, percent_str, percent
from collections import defaultdict

def default_scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def measure_neuron_cov(model, test_inputs, scaler, threshold=0, skip_layers=None, outs=None):
    if outs is None:
        outs = get_layer_outs_new(model, test_inputs, skip_layers)

    activation_table = defaultdict(bool)

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        for out_for_input in layer_out:  # out_for_input is output of layer for single input
            out_for_input = scaler(out_for_input)

            for neuron_index in range(out_for_input.shape[-1]):
                activation_table[(layer_index, neuron_index)] = activation_table[(layer_index, neuron_index)] or\
                                                                np.mean(out_for_input[..., neuron_index]) > threshold

    covered = len([1 for c in activation_table.values() if c])
    total = len(activation_table.keys())

    return percent_str(covered, total), covered, total, outs

from new_coverages.coverage import AbstractCoverage

class NeuronCoverage(AbstractCoverage):
    def __init__(self, model, scaler=default_scale, threshold=0, skip_layers=None):
        self.activation_table = defaultdict(bool)
        
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        
    def get_measure_state(self):
        return [self.activation_table]
    
    def set_measure_state(self, state):
        self.activation_table_by_section = state[0]

    def get_current_coverage(self):
        covered = len([1 for c in self.activation_table.values() if c])
        total = len(self.activation_table.keys())
        return percent(covered, total)
        
    def test(self, test_inputs):
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)

                for neuron_index in range(out_for_input.shape[-1]):
                    self.activation_table[(layer_index, neuron_index)] = self.activation_table[(layer_index, neuron_index)] or\
                                                                     np.mean(out_for_input[..., neuron_index]) > self.threshold

        covered = len([1 for c in self.activation_table.values() if c])
        total = len(self.activation_table.keys())

        return percent_str(covered, total), covered, total, outs
    

