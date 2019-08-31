from mnist_lenet_experiment import mnist_lenet_experiment
import numpy as np

from src.deephunter import DeepHunter

args, (train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser = mnist_lenet_experiment("random_multi_image")

#np.random.seed(seed=23)

test_input, _ = input_chooser()
print(test_input.shape)

coverage.step(test_images.reshape(-1,28,28,1))
print("initial coverage: %g" % (coverage.get_current_coverage()))

DeepHunter(test_images,coverage)