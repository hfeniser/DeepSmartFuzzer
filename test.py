import argparse

parser = argparse.ArgumentParser(description='Script for testing LeNet models for MNIST dataset using MCTS')
parser.add_argument("--lenet", type=int, default=1, choices=[1,4,5])
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
#from coverages.coverage import KMultisectionCoverage
#(state_len_calc_input, k, train_inputs) = (test_images[0].reshape(-1, 28, 28, 1), 10000, train_images.reshape(-1, 28, 28, 1))
#coverage = KMultisectionCoverage(model, state_len_calc_input, k, train_inputs)
from new_coverages.neuron_cov import NeuronCoverage
coverage = NeuronCoverage(model)

from coverages.coverage import NeuronCoverage
state_len_calc_input = test_images[0].reshape(-1, 28, 28, 1)
coverage2 = NeuronCoverage(model, state_len_calc_input)


np.random.seed(seed=213123)

from input_chooser import InputChooser
input_chooser = InputChooser(test_images, test_labels)
test1, _ = input_chooser()
test2, _ = input_chooser()

print("test1", test1[0][10])

print("test2", test2[0][10])

a = coverage.step(test1.reshape(-1,28,28,1), update_state=True)
b = coverage2.step(test1.reshape(-1,28,28,1), update_state=True)
print("1coverage: %g %g" % (coverage.get_current_coverage(), coverage2.get_current_coverage()))
print("1a: %g %g" % (a, b))



_, a = coverage.step(test2.reshape(-1,28,28,1), update_state=False)
_, b = coverage2.step(test2.reshape(-1,28,28,1), update_state=False)
print("2coverage: %g %g" % (coverage.get_current_coverage(), coverage2.get_current_coverage()))
print("2a: %g %g" % (a, b))

#from new_coverages.neuron_cov import NeuronCoverage
#coverage = NeuronCoverage(model)
#from coverages.coverage import NeuronCoverage
#coverage2 = NeuronCoverage(model, state_len_calc_input)
test = np.concatenate((test1, test2), axis=0)
print("test.shape", test.shape)
_, a = coverage.step(test.reshape(-1,28,28,1), update_state=False)
_, b = coverage2.step(test.reshape(-1,28,28,1), update_state=False)
print("3coverage: %g %g" % (coverage.get_current_coverage(), coverage2.get_current_coverage()))
print("3a: %g %g" % (a, b))