from mnist_lenet_experiment import mnist_lenet_experiment
import numpy as np
import itertools
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)
fig = plt.imshow(np.random.randint(0,256,size=(28,28)))
plt.figure(2)
fig2 = plt.imshow(np.random.randint(0,256,size=(28,28)))
plt.title("NOT FOUND ANY COVERAGE INCREASE")

(train_images, train_labels), (test_images, test_labels), model, coverage, input_chooser = mnist_lenet_experiment("random_one_image")

np.random.seed(seed=213123)

test_input, _ = input_chooser()
print(test_input.shape)

coverage.step(test_input.reshape(-1,28,28,1))
print("initial coverage: %g" % (coverage.get_current_coverage()))

input_lower_limit = 0
input_upper_limit = 255
action_division_p1 = (1,3,3,1)
actions_p2 = [-20, 20]

input_shape = test_input.shape
options_p1 = []
actions_p1_spacing = []
for i in range(len(action_division_p1)):
    spacing = int(input_shape[i] / action_division_p1[i])
    options_p1.append(list(range(0, input_shape[i], spacing)))
    actions_p1_spacing.append(spacing)

actions_p1 = list(itertools.product(*options_p1))

def apply_action(mutated_input, action1, action2):
    action_part1 = actions_p1[action1]
    action_part2 = actions_p2[action2]
    lower_limits = np.subtract(action_part1, actions_p1_spacing)
    lower_limits = np.clip(lower_limits, 0, action_part1) # lower_limits \in [0, action_part1]
    upper_limits = np.add(action_part1, actions_p1_spacing)
    upper_limits = np.clip(upper_limits, action_part1, input_shape) # upper_limits \in [action_part1, self.input_shape]
    s = tuple([slice(lower_limits[i], upper_limits[i]) for i in range(len(lower_limits))])
    mutated_input[s] += action_part2
    mutated_input[s] = np.clip(mutated_input[s], input_lower_limit, input_upper_limit)
    return mutated_input

def tc3(level, test_input, mutated_input):
    a1 = level > 10 # Tree Depth Limit
    a2 = not np.all(mutated_input >= 0) # Image >= 255
    a3 = not np.all(mutated_input <= 255) # Image <= 255
    a4 = not np.all(np.abs(mutated_input - test_input) < 80) # L_infinity < 20
    #if a3:
    #    index = np.where(mutated_input > 255)
    #    print(index, mutated_input[index])
    #print(a1, a2, a3, a4)
    return  a1 or a2 or a3 or a4

best_coverage, best_input = 0, np.copy(test_input)
iteration_count = 0
verbose, verbose_image = True, True

while not iteration_count > 100:
    level = 0
    mutated_input = np.copy(test_input)
    while not tc3(level, test_input, mutated_input):
        action1 = np.random.randint(0,len(actions_p1))
        action2 = np.random.randint(0,len(actions_p2))
        mutated_input = apply_action(mutated_input, action1, action2)
        level += 1
        _, coverage_sim = coverage.step(mutated_input, update_state=False)
        if verbose_image:
            plt.figure(1)
            fig.set_data(mutated_input.reshape((28,28)))
            plt.title("Action: " + str((action1,action2)) + " Coverage Increase: " + str(coverage_sim))
            plt.show()
            plt.pause(0.0001) #Note this correction
        #print("coverage", coverage_sim)
        if coverage_sim > best_coverage:
            best_input, best_coverage = np.copy(mutated_input), coverage_sim
            if verbose_image:
                plt.figure(2)
                fig2.set_data(best_input.reshape((28,28)))
                plt.title("BEST Coverage Increase: " + str(best_coverage))
                plt.show()
                plt.pause(0.0001) #Note this correction
    if verbose:
        print("Completed Iteration #%g" % (iteration_count))
        print("Current Coverage: %g" % (coverage_sim))
        print("Best Coverage up to now: %g" % (best_coverage))
    
    iteration_count += 1

print("found coverage increase", best_coverage)
print("found different input", np.any(best_input-test_input != 0))