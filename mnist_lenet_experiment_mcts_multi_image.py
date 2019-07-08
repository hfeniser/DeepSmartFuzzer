from mnist_lenet_experiment import mnist_lenet_experiment
import numpy as np

(train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser = mnist_lenet_experiment("mcts_multi_image")

np.random.seed(seed=213123)

test_input, _ = input_chooser()
coverage.step(test_images)
print("initial coverage: %g" % (coverage.get_current_coverage()))

# MCTS
from mcts import RLforDL_MCTS
input_lower_limit = 0
input_upper_limit = 255
action_division_p1 = (1,3,3,1)
actions_p2 = [-10, 10, ("translation", (10, 10)), ("rotation", 3), ("contrast", 1.2), ("blur", 1), ("blur", 4), ("blur", 7)]

def tc1(level, test_input, best_input, best_coverage): 
    # limit the level/depth of root
    return level > 8

def tc2(iterations):
    # limit the number of iterations on root
    return iterations > 20

def tc3(level, test_input, mutated_input):
    a1 = level > 6 # Tree Depth Limit
    a2 = not np.all(mutated_input >= 0) # Image >= 255
    a3 = not np.all(mutated_input <= 255) # Image <= 255
    a4 = not np.all(np.abs(mutated_input - test_input) < 100) # L_infinity < 20
    #if a3:
    #    index = np.where(mutated_input > 255)
    #    print(index, mutated_input[index])
    #print(a1, a2, a3, a4)
    return  a1 or a2 or a3 or a4

mcts = RLforDL_MCTS(test_input.shape, input_lower_limit, input_upper_limit,\
     action_division_p1, actions_p2, tc1, tc2, tc3)

for i in range(1, 1000):
    test_input, test_label = input_chooser()
    root, best_input, best_coverage = mcts.run(test_input, coverage)
    if best_coverage > 0:
        input_chooser.append(best_input, test_label)
        coverage.step(best_input, update_state=True)
        print("IMAGE %g SUCCEED" % (i))
        print("found coverage increase", best_coverage)
        print("found different input", np.any(best_input-test_input != 0))
        print("Current Total Coverage", coverage.get_current_coverage())
    else:
        print("IMAGE %g FAILED" % (i))