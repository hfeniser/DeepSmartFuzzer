import parent_import # adds parent directory to sys.path

import signal
import sys
def signal_handler(sig, frame):
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def mnist_lenet_experiment(model_name):
    import argparse

    parser = argparse.ArgumentParser(description=str(model_name) + ' experiment for testing LeNet models with MNIST dataset')
    parser.add_argument("--lenet", type=int, default=1, choices=[1,4,5])
    parser.add_argument("--coverage", type=str, default="neuron", choices=["neuron","kmn","nbc","snac"])
    parser.add_argument("--implicit_reward", type=bool, default=False)
    args = parser.parse_args()

    print("Arguments:", args)

    import numpy as np

    # MNIST DATASET
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(-1,28,28,1).astype(np.int16)
    test_images = test_images.reshape(-1,28,28,1).astype(np.int16)

    # LENET MODEL
    # multiple instance of openmp error 
    # solution: set environment variable
    # KMP_DUPLICATE_LIB_OK=TRUE
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    from LeNet.lenet_models import LeNet1, LeNet4, LeNet5
    from keras.layers import Input
    if args.lenet == 1:
        model = LeNet1(Input((28,28,1)))
    elif args.lenet == 4:
        model = LeNet4(Input((28,28,1)))
    elif args.lenet == 5:
        model = LeNet5(Input((28,28,1)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # COVERAGE
    if args.implicit_reward:
        def calc_implicit_reward_neuron(p1, p2):
            distance = np.abs(p1-p2)
            implicit_reward =  1 / (distance + 1)
            #print("p1, p2, distance, implicit_reward:", p1, p2, distance, implicit_reward)
            return implicit_reward

        def calc_implicit_reward(activation_values, covered_positions):
            #print("activation_values, covered_positions", activation_values, covered_positions)
            return np.max(activation_values * np.logical_not(covered_positions))
    else:
        calc_implicit_reward_neuron = None
        calc_implicit_reward = None

    if args.coverage == "neuron":
        from coverages.neuron_cov import NeuronCoverage
        coverage = NeuronCoverage(model, skip_layers=[0,5], calc_implicit_reward_neuron=calc_implicit_reward_neuron, calc_implicit_reward=calc_implicit_reward) # 0:input, 5:flatten
    elif args.coverage == "kmn" or args.coverage == "nbc" or args.coverage == "snac":
        from coverages.kmn import DeepGaugePercentCoverage
        k = 20
        coverage = DeepGaugePercentCoverage(model, k, train_images, skip_layers=[0,5], coverage_name=args.coverage, calc_implicit_reward_neuron=calc_implicit_reward_neuron, calc_implicit_reward=calc_implicit_reward) # 0:input, 5:flatten
    else:
        raise Exception("Unknown Coverage" + args.coverage)

    from src.input_chooser import InputChooser
    input_chooser = InputChooser(test_images, test_labels)

    return args, (train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser