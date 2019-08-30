from mnist_lenet_experiment import mnist_lenet_experiment
import numpy as np
import itertools

args, (train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser = mnist_lenet_experiment("mcts_multi_image")

#np.random.seed(seed=191)

test_input, _ = input_chooser()
coverage.step(test_images)
print("initial coverage: %g" % (coverage.get_current_coverage()))

# MCTS
from mcts import RLforDL_MCTS
input_lower_limit = 0
input_upper_limit = 255
action_division_p1 = (1,3,3,1)

translation = list(itertools.product(["translation"], [(10+10*k,10+10*k) for k in range(10)]))
scale = list(itertools.product(["scale"], [(1.5+0.5*k,1.5+0.5*k) for k in range(10)]))
shear = list(itertools.product(["shear"], [(-1.0+0.1*k,0) for k in range(10)]))
rotation = list(itertools.product(["rotation"], [3+3*k for k in range(10)]))
contrast = list(itertools.product(["contrast"], [1.2+0.2*k for k in range(10)]))
brightness = list(itertools.product(["brightness"], [10+10*k for k in range(10)]))
blur = list(itertools.product(["blur"], [k+1 for k in range(10)]))

actions_p2 = translation + rotation + contrast + brightness + blur

def tc1(level, test_input, best_input, best_coverage): 
    # limit the level/depth of root
    return level > 10

def tc2(iterations):
    # limit the number of iterations on root
    return iterations > 10

def tc3(level, test_input, mutated_input):
    c1 = level > 10 # Tree Depth Limit
    c2 = not np.all(np.abs(mutated_input - test_input) < 160) # L_infinity < 20
    return  c1 or c2

mcts = RLforDL_MCTS(test_input.shape, input_lower_limit, input_upper_limit,\
     action_division_p1, actions_p2, tc1, tc2, tc3, with_implicit_reward=args.implicit_reward, verbose_image=True)

for i in range(1, 1000):
    test_input, test_label = input_chooser(batch_size=64)
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