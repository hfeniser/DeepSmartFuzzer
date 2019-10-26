import numpy as np
import time


class Experiment:
    pass

def get_experiment(params):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment)
    experiment.model = _get_model(params, experiment)
    experiment.coverage = _get_coverage(params, experiment)
    experiment.input_chooser = _get_input_chooser(params, experiment)
    experiment.start_time = time.time()
    experiment.iteration = 0
    experiment.termination_condition = generate_termination_condition(experiment, params)
    return experiment

def generate_termination_condition(experiment, params):
    input_chooser = experiment.input_chooser
    nb_new_inputs = params.nb_new_inputs
    start_time = experiment.start_time
    time_period = params.time_period
    coverage = experiment.coverage
    nb_iterations = params.nb_iterations
    def termination_condition():
        c1 = len(input_chooser) - input_chooser.initial_nb_inputs > nb_new_inputs
        c2 = time.time() - start_time > time_period
        c3 = coverage.get_current_coverage() == 100
        c4 = nb_iterations is not None and experiment.iteration > nb_iterations
        return c1 or c2 or c3 or c4
    return termination_condition

def _get_dataset(params, experiment):
    if params.dataset == "MNIST":
        # MNIST DATASET
        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1).astype(np.int16)
        test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
    elif params.dataset == "CIFAR10":
        # CIFAR10 DATASET
        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        train_labels = train_labels.reshape(-1,)
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_labels = test_labels.reshape(-1,)
    else:
        raise Exception("Unknown Dataset:" + str(params.dataset))

    return {
        "train_inputs": train_images,
        "train_outputs": train_labels,
        "test_inputs": test_images,
        "test_outputs": test_labels
    }


def _get_model(params, experiment):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if params.model == "LeNet1":
        from src.LeNet.lenet_models import LeNet1
        from keras.layers import Input
        model = LeNet1(Input((28, 28, 1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "LeNet4":
        from src.LeNet.lenet_models import LeNet4
        from keras.layers import Input
        model = LeNet4(Input((28, 28, 1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "LeNet5":
        from src.LeNet.lenet_models import LeNet5
        from keras.layers import Input
        model = LeNet5(Input((28, 28, 1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "CIFAR_CNN":
        from keras.models import load_model
        model = load_model("src/CIFAR10/cifar_cnn.h5")
    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model


def _get_coverage(params, experiment):
    if not params.implicit_reward:
        params.calc_implicit_reward_neuron = None
        params.calc_implicit_reward = None

    # handle input scaling before giving input to model
    def input_scaler(test_inputs):
        model_lower_bound = params.model_input_scale[0]
        model_upper_bound = params.model_input_scale[1]
        input_lower_bound = params.input_lower_limit
        input_upper_bound = params.input_upper_limit
        scaled_input =  (test_inputs - input_lower_bound) / (input_upper_bound - input_lower_bound)
        scaled_input = scaled_input * (model_upper_bound - model_lower_bound) + model_lower_bound
        return scaled_input

    if params.coverage == "neuron":
        from coverages.neuron_cov import NeuronCoverage
        # TODO: Skip layers should be determined autoamtically
        coverage = NeuronCoverage(experiment.model, skip_layers=params.skip_layers,
                                  calc_implicit_reward_neuron=params.calc_implicit_reward_neuron,
                                  calc_implicit_reward=params.calc_implicit_reward)  # 0:input, 5:flatten
    elif params.coverage == "kmn" or params.coverage == "nbc" or params.coverage == "snac":
        from coverages.kmn import DeepGaugePercentCoverage
        # TODO: Skip layers should be determined autoamtically
        train_inputs_scaled = input_scaler(experiment.dataset["train_inputs"])
        coverage = DeepGaugePercentCoverage(experiment.model, getattr(params, 'kmn_k', 1000), train_inputs_scaled, skip_layers=params.skip_layers,
                                            coverage_name=params.coverage)  # 0:input, 5:flatten
    elif params.coverage == "tfc":
        from coverages.tfc import TFCoverage
        coverage = TFCoverage(experiment.model, params.tfc_subject_layer, params.tfc_threshold)
    else:
        raise Exception("Unknown Coverage" + str(params.coverage))

    coverage._step = coverage.step
    coverage.step = lambda test_inputs, *a, **kwa: coverage._step(input_scaler(test_inputs), *a, **kwa)

    return coverage


def _get_input_chooser(params, experiment):
    if params.input_chooser == "random":
        from src.input_chooser import InputChooser
        input_chooser = InputChooser(experiment.dataset["test_inputs"], experiment.dataset["test_outputs"])
    elif params.input_chooser == "clustered_random":
        from src.clustered_input_chooser import ClusteredInputChooser
        input_chooser = ClusteredInputChooser(experiment.dataset["test_inputs"], experiment.dataset["test_outputs"])
    else:
        raise Exception("Unknown Input Chooser" + str(params.input_chooser))

    return input_chooser
