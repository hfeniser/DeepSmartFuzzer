import sys
sys.path.append('../')

import numpy as np
from coverages.utils import get_layer_outs_new, percent_str, percent
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

from coverages.coverage import AbstractCoverage

class NeuronCoverage(AbstractCoverage):
    def __init__(self, model, scaler=default_scale, threshold=0.75, skip_layers=None, calc_implicit_reward_neuron=None, calc_implicit_reward=None):
        self.activation_table = defaultdict(float)
        
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.calc_implicit_reward_neuron = calc_implicit_reward_neuron
        self.calc_implicit_reward = calc_implicit_reward
        
    def get_measure_state(self):
        return [self.activation_table]
    
    def set_measure_state(self, state):
        self.activation_table = state[0]

    def reset_measure_state(self):
        self.activation_table = defaultdict(float)

    def get_current_coverage(self, with_implicit_reward=False):
        if len(self.activation_table.keys()) == 0:
            return 0

        reward, covered, implicit_reward = self.calc_reward(self.activation_table, with_implicit_reward=with_implicit_reward)
        total = len(self.activation_table.keys())
        #print("covered, implicit_reward", covered, implicit_reward)
        return percent(reward, total)
        
    def test(self, test_inputs, with_implicit_reward=False):
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                out_for_input = self.scaler(out_for_input)

                for neuron_index in range(out_for_input.shape[-1]):
                    if self.activation_table[(layer_index, neuron_index)] == 1:
                        pass
                    elif np.mean(out_for_input[..., neuron_index]) > self.threshold:
                        self.activation_table[(layer_index, neuron_index)] =  1
                    elif self.calc_implicit_reward_neuron:
                        p1 = np.mean(out_for_input[..., neuron_index])
                        p2 = self.threshold
                        r = self.calc_implicit_reward_neuron(p1, p2)
                        self.activation_table[(layer_index, neuron_index)] = r                           

        reward, covered, implicit_reward = self.calc_reward(self.activation_table, with_implicit_reward=with_implicit_reward)
        total = len(self.activation_table.keys())
        return percent_str(reward, total), reward, total, outs
