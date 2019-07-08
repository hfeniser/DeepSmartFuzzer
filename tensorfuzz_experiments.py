# 2018. Augustus Odena, Ian Goodfellow. TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing

import argparse

parser = argparse.ArgumentParser(description='Script for testing LeNet models for MNIST dataset using TensorFuzz')
parser.add_argument("--lenet", type=int, default=1, choices=[1,4,5])
parser.add_argument("--coverage", type=str, default="kmn", choices=["neuron","kmn"])
args = parser.parse_args()

print("Arguments:", args)

import os, sys
try:
    import tensorfuzz.lib
except:    
    os.system("git clone https://github.com/brain-research/tensorfuzz.git")
    try:
        import tensorfuzz.lib
        print("TensorFuzz Installed")
    except:
        raise Exception("TensorFuzz Installation Failed")

