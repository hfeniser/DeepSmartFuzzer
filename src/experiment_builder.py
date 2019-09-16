import numpy as np

class Experiment:
    pass

def get_experiment(params):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment)
    experiment.model = _get_model(params, experiment)
    experiment.coverage = _get_coverage(params, experiment)
    experiment.input_chooser = _get_input_chooser(params, experiment)
    return experiment


def _get_dataset(params, experiment):
    if params.dataset == "MNIST":
        # MNIST DATASET
        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(-1,28,28,1).astype(np.int16)
        test_images = test_images.reshape(-1,28,28,1).astype(np.int16)
    elif params.dataset == "CIFAR10":
        # CIFAR10 DATASET
        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
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
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    if params.model == "LeNet1":
        from src.LeNet.lenet_models import LeNet1
        from keras.layers import Input
        model = LeNet1(Input((28,28,1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Lenet4":
        from src.LeNet.lenet_models import LeNet4
        from keras.layers import Input
        model = LeNet4(Input((28,28,1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "Lenet5":
        from src.CIFAR10LeNet.lenet_models import LeNet5
        from keras.layers import Input
        model = LeNet5(Input((28,28,1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "CIFAR_ORIGINAL":
        from keras.models import load_model
        model = load_model("src/CIFAR10/cifar_original.h5")
    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model

def _get_coverage(params, experiment):
    if not params.implicit_reward:
        params.calc_implicit_reward_neuron = None
        params.calc_implicit_reward = None

    if params.coverage == "neuron":
        from coverages.neuron_cov import NeuronCoverage
        coverage = NeuronCoverage(experiment.model, skip_layers=[0,5], calc_implicit_reward_neuron=params.calc_implicit_reward_neuron, calc_implicit_reward=params.calc_implicit_reward) # 0:input, 5:flatten
    elif params.coverage == "kmn" or params.coverage == "nbc" or params.coverage == "snac":
        from coverages.kmn import DeepGaugePercentCoverage
        k = 20
        coverage = DeepGaugePercentCoverage(experiment.model, k, experiment.dataset["train_inputs"], skip_layers=[0,5], coverage_name=params.coverage, calc_implicit_reward_neuron=params.calc_implicit_reward_neuron, calc_implicit_reward=params.calc_implicit_reward) # 0:input, 5:flatten
    else:
        raise Exception("Unknown Coverage" + str(params.coverage))

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