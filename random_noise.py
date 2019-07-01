import argparse

parser = argparse.ArgumentParser(description='Script for testing LeNet models for MNIST dataset using random noise')
parser.add_argument("--lenet", type=int, default=1, choices=[1,4,5])
args = parser.parse_args()

print("Arguments:", args)

import numpy as np

import matplotlib.pyplot as plt
plt.ion()
fig = plt.imshow(np.random.randint(0,256,size=(28,28)))

import signal
import sys
def signal_handler(sig, frame):
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

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
from coverages.neuron_cov import NeuronCoverage
coverage = NeuronCoverage(model)


np.random.seed(seed=213123)

from input_chooser import InputChooser
input_chooser = InputChooser(test_images, test_labels)
test_input, _ = input_chooser()
print(test_input.shape)

coverage.step(test_input.reshape(-1,28,28,1))
print("initial coverage: %g" % (coverage.get_current_coverage()))

mu, sigma = 20, 10 
input_lower_limit, input_upper_limit = 0, 255

best_input_sim, best_coverage_sim = np.copy(test_input), 0

for i in range(1000):
    mutated_input = np.copy(test_input)
    x = np.array(mu + sigma * np.random.randn(1,28,28,1), dtype=int)
    mutated_input += x
    mutated_input = np.clip(mutated_input, input_lower_limit, input_upper_limit)

    _, coverage_sim = coverage.step(test_input.reshape(-1,28,28,1), update_state=False)

    if coverage_sim > best_coverage_sim:
        best_input_sim, best_coverage_sim = np.copy(mutated_input), coverage_sim
    
    fig.set_data(mutated_input.reshape((28,28)))
    plt.title("iteration: " + str(i) + ", coverage increase: " + str(coverage_sim))
    plt.show()
    plt.pause(0.0001) #Note this correction

    print("Completed Iteration #%g" % (i))
    print("Current Coverage From Simulation: %g" % (coverage_sim))
    print("Best Coverage up to now: %g" % (best_coverage_sim))


print("found coverage increase", best_coverage_sim)
print("found different input", np.any(best_input_sim-test_input != 0))