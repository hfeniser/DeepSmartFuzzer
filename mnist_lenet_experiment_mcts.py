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
from coverages.coverage import NeuronCoverage
state_len_calc_input = test_images[0].reshape(-1, 28, 28, 1)
coverage = NeuronCoverage(model, state_len_calc_input)


np.random.seed(seed=213123)

from input_chooser import InputChooser
input_chooser = InputChooser(test_images, test_labels)
test_input, _ = input_chooser()
print(test_input.shape)

coverage.step(test_input.reshape(-1,28,28,1))
print("initial coverage: %g" % (coverage.get_current_coverage()))

# MCTS
from mcts import RLforDL_MCTS

actions_p1_spacing = (1,3,3,1)
actions_p2 = [-1, +1]

def tc1(level, test_input, best_input, best_coverage): 
    # limit the level/depth of root
    return level > 10

def tc2(iterations):
    # limit the number of iterations on root
    return iterations > 10000

def tc3(level, test_input, mutated_input):
    a1 = level > 10 # Tree Depth Limit
    a2 = not (np.all(mutated_input <= 255) and np.all(mutated_input >= 0)) # Image [0,255]
    a3 = not np.all(np.abs(mutated_input - test_input) < 20) # L_infinity < 20
    #print(a1, a2, a3)
    return  a1 or a2 or a3

mcts = RLforDL_MCTS(test_input.shape, actions_p1_spacing, actions_p2, tc1, tc2, tc3)
root, best_input, best_coverage = mcts.run(test_input, coverage)
print("found coverage increase", best_coverage)
print("found different input", np.any(best_input-test_input != 0))