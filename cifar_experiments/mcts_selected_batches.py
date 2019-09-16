from mnist_lenet_experiment import mnist_lenet_experiment
import numpy as np
import itertools

args, (train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser = mnist_lenet_experiment("mcts_multi_image")

#np.random.seed(seed=191)

test_input, _ = input_chooser()
coverage.step(test_images)
print("initial coverage: %g" % (coverage.get_current_coverage()))

# MCTS
from src_v2.mcts import MCTS_Node, run_mcts
from src_v2.RLforDL import RLforDL, RLforDL_State, Reward_Status

input_lower_limit = 0
input_upper_limit = 255
action_division_p1 = (1,3,3,1)

translation = list(itertools.product(["translation"], [(-5,-5), (-5,0), (0,-5), (0,0), (5,0), (0,5), (5,5)]))
rotation = list(itertools.product(["rotation"], [-15,-12,-9,-6,-3,3,6,9,12,15]))
contrast = list(itertools.product(["contrast"], [1.2+0.2*k for k in range(10)]))
brightness = list(itertools.product(["brightness"], [10+10*k for k in range(10)]))
blur = list(itertools.product(["blur"], [k+1 for k in range(10)]))

actions_p2 = contrast + brightness + blur

def tc1(state): 
    # limit the level/depth of root
    return state.level > 8

def tc2(iterations):
    # limit the number of iterations on root
    return iterations > 25

def tc3(state):
    original_input = state.original_input
    mutated_input = state.mutated_input

    alpha, beta = 0.1, 0.5
    if(np.sum((original_input-mutated_input) != 0) < alpha * np.sum(original_input>0)):
        return not np.max(np.abs(mutated_input-original_input)) <= 255
    else:
        return not np.max(np.abs(mutated_input-original_input)) <= beta*255

game = RLforDL(coverage, test_input.shape, input_lower_limit, input_upper_limit,\
     action_division_p1, actions_p2, tc3, with_implicit_reward=args.implicit_reward)

import glob, os

fileList = glob.glob('data/mcts*', recursive=True)
for f in fileList:
    os.remove(f)

for i in range(0, 30):
    test_input = np.load("data/deephunter_{}.npy".format((i%10)+1))
    root_state = RLforDL_State(test_input, 0, game=game)
    root = MCTS_Node(root_state, game)
    run_mcts(root, tc1, tc2)
    best_coverage, best_input = game.get_stat()
    game.reset_stat()
    if best_coverage > 0:
        coverage.step(best_input, update_state=True)
        np.save("data/mcts_{}.npy".format(i+1), best_input)
        print("IMAGE %g SUCCEED" % (i))
        print("found coverage increase", best_coverage)
        print("found different input", np.any(best_input-test_input != 0))
        print("Current Total Coverage", coverage.get_current_coverage())
    else:
        print("IMAGE %g FAILED" % (i))