import argparse
import importlib
from src.utility import str2bool, merge_object
from src.experiment_builder import get_experiment

import signal
import sys


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def run_experiment(params):
    params = load_params(params)
    experiment = get_experiment(params)

    if params.verbose:
        print("Parameters:", params)

    experiment.coverage.step(experiment.dataset["test_inputs"])

    if params.verbose:
        print("initial coverage: %g" % (experiment.coverage.get_current_coverage()))

    experiment.runner = load_runner(params)
    experiment.runner(params, experiment)


def load_params(params):
    m = importlib.import_module("params." + params.params_set)
    new_params = getattr(m, params.params_set)
    return merge_object(params, new_params)


def load_runner(params):
    m = importlib.import_module("runners." + params.runner)
    runner = getattr(m, params.runner)
    return runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments Script For RLforDL")
    parser.add_argument("--params_set", type=str, default="mnist_lenet",
                        choices=["mnist_lenet", "cifar10", "deephunter_mnist"])
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--model", type=str, default="LeNet1", choices=["LeNet1", "LeNet4", "LeNet5", "CIFAR_ORIGINAL"])
    parser.add_argument("--implicit_reward", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--coverage", type=str, default="neuron", choices=["neuron", "kmn", "nbc", "snac", "tfc"])
    parser.add_argument("--input_chooser", type=str, default="random", choices=["random", "clustered_random"])
    parser.add_argument("--runner", type=str, default="mcts",
                        choices=["mcts", "mcts_clustered", "mcts_selected", "deephunter", "tensorfuzz"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nb_iterations", type=int, default=30)
    parser.add_argument("--verbose", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--image_verbose", type=str2bool, nargs='?', const=True, default=True)
    params = parser.parse_args()

    run_experiment(params)
