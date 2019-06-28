# -*- coding: utf-8 -*-
import numpy as np
from utils import get_layer_outs, get_layer_outs_new, percent_str


def measure_tkn_old(model, test_inputs, k):
    activation_table = {}
    neuron_count = 0
    outs = get_layer_outs(model, test_inputs)

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        for (index, out_for_input) in enumerate(layer_out[0]):  # out_for_input is output of layer for single input
            if index == 0:
                neuron_count += out_for_input.size

            top_k_neuron_indexes = np.argsort(out_for_input, axis=None)[-k:len(out_for_input)]

            for neuron_index in top_k_neuron_indexes:
                activation_table[(layer_index, neuron_index)] = True

    covered = len(activation_table.keys())

    return covered, neuron_count, float(covered) / neuron_count * 100


def measure_tkn(model, test_inputs, k, skip=None):
    if skip is None:
        skip = []

    activation_table = {}
    neuron_count_by_layer = {}
    outs = get_layer_outs(model, test_inputs, skip=skip)

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        for (index, out_for_input) in enumerate(layer_out[0]):  # out_for_input is output of layer for single input
            neuron_outs = np.zeros((out_for_input.shape[-1],))
            neuron_count_by_layer[layer_index] = len(neuron_outs)

            for i in range(out_for_input.shape[-1]):
                neuron_outs[i] = np.mean(out_for_input[..., i])

            top_k_neuron_indexes = np.argsort(neuron_outs, axis=None)[-k:len(neuron_outs)]

            for neuron_index in top_k_neuron_indexes:
                activation_table[(layer_index, neuron_index)] = True

    neuron_count = sum(neuron_count_by_layer.values())
    covered = len(activation_table.keys())

    return covered, neuron_count, float(covered) / neuron_count * 100


def measure_tkn_with_pattern(model, test_inputs, k, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_inputs, skip=skip)[:-1]

    activation_table = {}
    neuron_count_by_layer = {}
    pattern_set = set()

    layer_count = len(outs)
    print(layer_count)
    for input_index in range(len(test_inputs)):  # out_for_input is output of layer for single input
        pattern = []

        for layer_index in range(layer_count):  # layer_out is output of layer for all inputs
            out_for_input = outs[layer_index][0][input_index]

            neuron_outs = np.zeros((out_for_input.shape[-1],))
            neuron_count_by_layer[layer_index] = len(neuron_outs)
            for i in range(out_for_input.shape[-1]):
                neuron_outs[i] = np.mean(out_for_input[..., i])

            top_k_neuron_indexes = (np.argsort(neuron_outs, axis=None)[-k:len(neuron_outs)])
            pattern.append(tuple(top_k_neuron_indexes))

            for neuron_index in top_k_neuron_indexes:
                activation_table[(layer_index, neuron_index)] = True

            if layer_index + 1 == layer_count:
                pattern_set.add(tuple(pattern))
    neuron_count = sum(neuron_count_by_layer.values())
    covered = len(activation_table.keys())
    print(neuron_count)
    print(covered)
    return percent_str(covered, neuron_count), covered, neuron_count, len(pattern_set), outs


class DeepGaugeLayerLevelCoverage:
    def __init__(self, model, k, skip_layers=None):
        self.activation_table = {}
        self.pattern_set = set()
        
        self.model = model
        self.k = k
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        
    def get_measure_state(self):
        return [self.activation_table, self.pattern_set]
    
    def set_measure_state(self, state):
        self.activation_table = state[0]
        self.pattern_set = state[1]
        
    def test(self, test_inputs):
        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        
        neuron_count_by_layer = {}

        layer_count = len(outs)

        for input_index in range(len(test_inputs)):  # out_for_input is output of layer for single input
            pattern = []

            for layer_index in range(layer_count):  # layer_out is output of layer for all inputs
                out_for_input = outs[layer_index][input_index]

                neuron_outs = np.zeros((out_for_input.shape[-1],))
                neuron_count_by_layer[layer_index] = len(neuron_outs)
                for i in range(out_for_input.shape[-1]):
                    neuron_outs[i] = np.mean(out_for_input[..., i])

                top_k_neuron_indexes = (np.argsort(neuron_outs, axis=None)[-self.k:len(neuron_outs)])
                pattern.append(tuple(top_k_neuron_indexes))

                for neuron_index in top_k_neuron_indexes:
                    self.activation_table[(layer_index, neuron_index)] = True

                if layer_index + 1 == layer_count:
                    self.pattern_set.add(tuple(pattern))
                    
        neuron_count = sum(neuron_count_by_layer.values())
        covered = len(self.activation_table.keys())
        
        return (percent_str(covered, neuron_count), # tknc
                covered, neuron_count,
                len(self.pattern_set), # tknp
                outs)
