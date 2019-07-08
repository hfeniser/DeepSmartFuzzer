
import signal
import sys
def signal_handler(sig, frame):
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def mnist_lenet_experiment(model_name):
    import argparse

    parser = argparse.ArgumentParser(description=str(model_name) + ' experiment for testing LeNet models with MNIST dataset')
    parser.add_argument("--lenet", type=int, default=1, choices=[1,4,5])
    parser.add_argument("--coverage", type=str, default="neuron", choices=["neuron","kmn"])
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

    from models.lenet_models import LeNet1, LeNet4, LeNet5
    from keras.layers import Input
    if args.lenet == 1:
        model = LeNet1(Input((28,28,1)))
    elif args.lenet == 4:
        model = LeNet4(Input((28,28,1)))
    elif args.lenet == 5:
        model = LeNet5(Input((28,28,1)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # COVERAGE
    if args.coverage == "neuron":
        from coverages.neuron_cov import NeuronCoverage
        coverage = NeuronCoverage(model)
    elif args.coverage == "kmn":
        from coverages.kmn import DeepGaugePercentCoverage
        k = 20
        coverage = DeepGaugePercentCoverage(model, k, train_images)
    else:
        raise Exception("Unknown Coverage" + args.coverage)

    from input_chooser import InputChooser
    input_chooser = InputChooser(test_images, test_labels)

    return (train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser
         