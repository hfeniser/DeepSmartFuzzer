import argparse
import importlib
import numpy as np
import random
import time
from src.utility import str2bool, merge_object
from src.experiment_builder import get_experiment
from src.adversarial import check_adversarial
from matplotlib.pyplot import imsave 
import os
import shutil

import signal
import sys
def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def run_experiment(params):
    params = load_params(params)
    experiment = get_experiment(params)

    if params.random_seed is not None:
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)


    if params.verbose:
        print("Parameters:", params)

    experiment.coverage.step(experiment.dataset["test_inputs"])

    inital_coverage = experiment.coverage.get_current_coverage()
    if params.verbose:
        print("initial coverage: %g" % (inital_coverage))

    experiment.runner = load_runner(params)
    experiment.runner(params, experiment)
    
    final_coverage = experiment.coverage.get_current_coverage()
    if params.verbose:
        print("initial coverage: %g" % (inital_coverage))
        time_passed_min = (time.time() - experiment.start_time) / 60
        print("time passed (minutes): %g" % time_passed_min)
        print("iterations: %g" % experiment.iteration)
        print("number of new inputs: %g" % (len(experiment.input_chooser) - experiment.input_chooser.initial_nb_inputs))
        print("final coverage: %g" % (final_coverage))
        print("total coverage increase: %g" % (final_coverage - inital_coverage))

    if params.check_adversarial:
        check_adversarial(experiment, params)

    if params.save_generated_samples:
        i = experiment.input_chooser.initial_nb_inputs
        if params.input_chooser == "clustered_random":
            new_inputs = experiment.input_chooser.test_inputs[i:]
            new_outputs = experiment.input_chooser.test_outputs[i:]
        else:
            new_inputs = experiment.input_chooser.features[i:]
            new_outputs = experiment.input_chooser.labels[i:]

        shutil.rmtree("generated_samples", ignore_errors=True)
        os.makedirs("generated_samples")
        if new_inputs.shape[-1] == 1:
            new_inputs = new_inputs.reshape(new_inputs.shape[:-1])
        for i in range(len(new_inputs)):
            imsave('generated_samples/%g.png'%i, new_inputs[i])
        np.save('generated_samples/new_inputs', new_inputs)
        np.save('generated_samples/new_outputs', new_outputs)

def load_params(params):
    for params_set in params.params_set:
        m = importlib.import_module("params." + params_set)
        new_params = getattr(m, params_set)
        params = merge_object(params, new_params)
    return params
    
def load_runner(params):
    m = importlib.import_module("runners." + params.runner)
    runner = getattr(m, params.runner)
    return runner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments Script For RLforDL")
    parser.add_argument("--params_set", nargs='*', type=str, default=["mnist", "mcts", "tfc"], help="see params folder")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--model", type=str, default="LeNet1", choices=["LeNet1", "LeNet4", "LeNet5", "CIFAR_CNN"])
    parser.add_argument("--implicit_reward", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--coverage", type=str, default="neuron", choices=["neuron", "kmn", "nbc", "snac", "tfc"])
    parser.add_argument("--input_chooser", type=str, default="random", choices=["random", "clustered_random"])
    parser.add_argument("--runner", type=str, default="mcts", choices=["mcts", "mcts_clustered", "deephunter", "tensorfuzz"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nb_iterations", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--image_verbose", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--check_adversarial", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--save_generated_samples", type=str2bool, nargs='?', const=True, default=False)
    params = parser.parse_args()

    run_experiment(params)

