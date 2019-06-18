# Sign-sign cover
import sys
sys.path.append('../')

from utils import get_layer_outs, percent_str


def default_sign_fn(x, _=None):
    return +1 if x > 0 else -1


# Optimized implementation
def measure_ss_cover(model, test_inputs, sign_fn=default_sign_fn, skip_layers=None, outs=None):
    if outs is None:
        outs = get_layer_outs(model, test_inputs, skip_layers)

    cover_set = set()

    total_pairs = 0

    for layer_index in range(len(outs) - 1):
        lower_layer_outs, upper_layer_outs = outs[layer_index][0], outs[layer_index + 1][0]
        lower_layer_fn, upper_layer_fn = model.layers[layer_index], model.layers[layer_index + 1]

        print(layer_index)

        for lower_neuron_index in range(lower_layer_outs.shape[-1]):
            for upper_neuron_index in range(upper_layer_outs.shape[-1]):
                total_pairs += 1

                sign_set = set()

                for input_index in range(len(test_inputs)):
                    rest_signs = \
                        tuple(sign_fn(lower_layer_outs[input_index][i], lower_layer_fn)
                              for i in range(lower_layer_outs.shape[-1]) if i != lower_neuron_index)

                    all_signs = (sign_fn(lower_layer_outs[input_index][lower_neuron_index], lower_layer_fn),
                                 sign_fn(upper_layer_outs[input_index][upper_neuron_index], upper_layer_fn),
                                 rest_signs)

                    if (-1 * all_signs[0], -1 * all_signs[1], all_signs[2]) in sign_set:
                        cover_set.add(((layer_index, lower_neuron_index), (layer_index + 1, upper_neuron_index)))

                    sign_set.add(all_signs)

    covered_pair_count = len(cover_set)

    return percent_str(covered_pair_count, total_pairs), covered_pair_count, total_pairs, cover_set, outs


# Reference implementation
def measure_ss_cover_naive(model, test_inputs, sign_fn=default_sign_fn, skip_layers=None):
    if skip_layers is None:
        skip_layers = []

    cover_set = set()
    outs = get_layer_outs(model, test_inputs, skip_layers)
    total_pairs = 0

    for layer_index in range(len(outs) - 1):
        lower_layer_outs, upper_layer_outs = outs[layer_index][0], outs[layer_index + 1][0]
        lower_layer_fn, upper_layer_fn = model.layers[layer_index], model.layers[layer_index + 1]

        print(layer_index)

        for lower_neuron_index in range(lower_layer_outs.shape[-1]):
            for upper_neuron_index in range(upper_layer_outs.shape[-1]):
                total_pairs += 1

                for input_index_i in range(len(test_inputs)):
                    for input_index_j in range(input_index_i + 1, len(test_inputs)):

                        covered = (sign_fn(lower_layer_outs[input_index_i][lower_neuron_index], lower_layer_fn) !=
                                   sign_fn(lower_layer_outs[input_index_j][lower_neuron_index], lower_layer_fn))

                        covered = covered and \
                            (sign_fn(upper_layer_outs[input_index_i][upper_neuron_index], upper_layer_fn) !=
                             sign_fn(upper_layer_outs[input_index_j][upper_neuron_index], upper_layer_fn))

                        for other_lower_neuron_index in range(lower_layer_outs.shape[-1]):
                            if other_lower_neuron_index == lower_neuron_index:
                                continue

                            covered = covered and \
                                (sign_fn(lower_layer_outs[input_index_i][other_lower_neuron_index], lower_layer_fn) ==
                                 sign_fn(lower_layer_outs[input_index_j][other_lower_neuron_index], lower_layer_fn))

                        if covered:
                            cover_set.add(((layer_index, lower_neuron_index), (layer_index + 1, upper_neuron_index)))

    covered_pair_count = len(cover_set)

    return covered_pair_count, total_pairs, cover_set
