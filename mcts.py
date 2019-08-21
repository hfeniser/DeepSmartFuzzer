import numpy as np
import itertools

import image_transforms


distance_in_reward = False

import matplotlib.pyplot as plt
rows, columns = 8, 8
plt.ion()
fig=plt.figure(1, figsize=(8, 8))
fig_plots = []
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    subplot = plt.imshow(np.random.randint(0,256,size=(28,28)))
    fig_plots.append(subplot)
plt.show()
fig2=plt.figure(2, figsize=(8, 8))
fig2.suptitle("NOT FOUND ANY COVERAGE INCREASE")
fig2_plots = []
for i in range(1, columns*rows +1):
    fig2.add_subplot(rows, columns, i)
    subplot = plt.imshow(np.random.randint(0,256,size=(28,28)))
    fig2_plots.append(subplot)
plt.show()

import signal
import sys
def signal_handler(sig, frame):
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

class MCTS_Node:
    def __init__(self, nb_child_nodes, state):
        self.parent = None
        self.level = 0
        self.value = 0
        self.relative_index = 0
        self.visit_count = 1
        self.child_nodes = [None] * nb_child_nodes
        self.state = state

    def potential(self, C=np.sqrt(2)):
        # Upper Confidence Bound (UCB)
        # C: hyperparammer
        # See following for details on UCB:
        # 2008. G.M.J.B. Chaslot et all. Progressive strategies for monte-carlo tree search.
        # 2006. Levente Kocsis et all. Bandit based monte-carlo planning.
        return self.value/self.visit_count + C*np.sqrt(np.log(self.parent.visit_count)/self.visit_count)

    def selection(self, C=np.sqrt(2)):
        p = np.array([child.potential(C) for child in self.child_nodes])
        #print(" -float((p.min() < 0)*p.min())",  -float((p.min() < 0)*p.min()))
        #p += -float((p.min() < 0)*p.min())
        p /= p.sum()
        return self.child_nodes[np.random.choice(range(len(self.child_nodes)), p=p)]

    def expansion(self, child_index, new_node):
        new_node.parent = self
        new_node.level = self.level+1
        new_node.relative_index = child_index
        self.child_nodes[child_index] = new_node

    def backprop(self, reward):
        self.value += reward
        print("self.level, self.value", self.level, self.value)
        if reward != 0 and self.parent != None:
            self.parent.backprop(reward)

    def isLeaf(self):
        return self.child_nodes.count(None) == len(self.child_nodes)

    def bestChild(self, C=np.sqrt(2)):
        best = self.child_nodes[0]
        best_r = best.value / best.visit_count

        print("x1")

        for i in range(len(self.child_nodes)):
            if self.child_nodes[i].visit_count == 0:
                current_r = 0
            else:
                current_r = self.child_nodes[i].value / self.child_nodes[i].visit_count
            print("x2", i, len(self.child_nodes))
            if best_r < current_r:
                best = self.child_nodes[i]
                best_r = current_r
        
        if best_r == 0:
            print("x3", [self.child_nodes[i].value for i in range(len(self.child_nodes))])
            print("simulations failed to find any reward")
            raise Exception("simulations failed to find any reward")
        return best

    def printPath(self, end="\n"):
        if self.parent:
            self.parent.printPath(end="->")
            print(self.relative_index, end=end)
        else:
            print("root", end=end)


class RLforDL_MCTS_State:
    def __init__(self, mutated_input):
        self.mutated_input = mutated_input

def find_the_distance(mutated_input, last_node):
    # find root node
    root = last_node
    while(root.parent != None):
        root = root.parent

    # get the initial input from the root node
    initial_input = root.state.mutated_input

    # calc distance
    dist = np.sum((mutated_input - initial_input)**2) / mutated_input.size
    #print("dist", dist)
    return dist


class RLforDL_MCTS:
    def __init__(self, input_shape, input_lower_limit, input_upper_limit, action_division_p1, actions_p2, tc1, tc2, tc3, with_implicit_reward=False, verbose=True, verbose_image=True):
        self.input_shape = input_shape
        self.input_lower_limit = input_lower_limit
        self.input_upper_limit = input_upper_limit
        options_p1 = []
        self.actions_p1_spacing = []
        for i in range(len(action_division_p1)):
            spacing = int(self.input_shape[i] / action_division_p1[i])
            options_p1.append(list(range(0, self.input_shape[i], spacing)))
            self.actions_p1_spacing.append(spacing)

        self.actions_p1 = list(itertools.product(*options_p1))
        self.actions_p2 = actions_p2
        self.tc1 = tc1 # termination condition for the entire program e.g. limit the number of epochs
        self.tc2 = tc2 # termination condition for the current iteration e.g. limit the iterations
        self.tc3 = tc3 # cut-off condition for the tree
        self.with_implicit_reward = with_implicit_reward
        self.verbose = verbose
        self.verbose_image = verbose_image
        if self.verbose:
            print("self.actions_p1", self.actions_p1)
            print("self.actions_p1_spacing", self.actions_p1_spacing)
            print("self.actions_p2", self.actions_p2)
    
    def player(self, level):
        if level % 2 == 0:
            return 1
        else:
            return 2

    def player_for_node(self, node):
        return self.player(node.level)

    def apply_action(self, mutated_input, action1, action2):
        action_part1 = self.actions_p1[action1]
        action_part2 = self.actions_p2[action2]
        #print("action:", action_part1, action_part2)
        lower_limits = np.subtract(action_part1, self.actions_p1_spacing)
        lower_limits = np.clip(lower_limits, 0, action_part1) # lower_limits \in [0, action_part1]
        upper_limits = np.add(action_part1, self.actions_p1_spacing)
        upper_limits = np.clip(upper_limits, action_part1, self.input_shape) # upper_limits \in [action_part1, self.input_shape]
        s = tuple([slice(lower_limits[i], upper_limits[i]) for i in range(len(lower_limits))])
        for j in range(len(mutated_input)):
            mutated_input_piece = mutated_input[j].reshape(self.input_shape)
            if not isinstance(action_part2, tuple):
                mutated_input_piece[s] += action_part2
            else:
                f = getattr(image_transforms,'image_'+action_part2[0])
                m_shape = mutated_input_piece[s].shape
                i = mutated_input_piece[s].reshape(m_shape[-3:])
                i = f(i, action_part2[1])
                mutated_input_piece[s] = i.reshape(m_shape)
            mutated_input_piece[s] = np.clip(mutated_input_piece[s], self.input_lower_limit, self.input_upper_limit)
        return mutated_input

    def apply_action_for_node(self, node, child_index):
        # Player 1: select a position in the input (e.g. pixel for image)
        # Player 2: select a manipulation on the selected region (e.g. +5,-5 for image)
        # action is complete when player 2 done
        mutated_input = np.copy(node.state.mutated_input)
        if self.player_for_node(node) == 2:
            mutated_input = self.apply_action(mutated_input, node.relative_index, child_index)

        return mutated_input

    def simulate_for_node(self, node, test_input, coverage):
        # Simulation
        level = node.level
        input_sim = np.copy(node.state.mutated_input)

        if self.player(level) == 1:
            action1 = np.random.randint(0,len(self.actions_p1))
            action2 = np.random.randint(0,len(self.actions_p2))
            input_sim = self.apply_action(input_sim, action1, action2)
            level += 2
        else:
            action1 = node.relative_index
            action2 = np.random.randint(0,len(self.actions_p2))
            input_sim = self.apply_action_for_node(node, action2)
            level += 1
        
        if self.verbose_image:
            plt.figure(1)
            fig.suptitle("level:" + str(level) + " Action: " + str((action1,action2)))
            for i in range(len(input_sim[0:64])):
                fig_plots[i].set_data(input_sim[i].reshape((28,28)))
            fig.canvas.flush_events()

        if self.tc3(level, test_input, input_sim):
            # already an termination node
            print("termination node")
            return input_sim, 0
        
        _, reward = coverage.step(input_sim, update_state=False, with_implicit_reward=self.with_implicit_reward)
        print("reward", reward)
        if distance_in_reward:
            dist = find_the_distance(input_sim, node)
            if reward > 0 and dist > 0:
                reward = reward / dist
        return input_sim, reward, action1, action2

    def run(self, test_input, coverage, C=np.sqrt(2)):
        best_input, best_coverage = np.copy(test_input), 0
        print("batch shape:", test_input.shape)
        root = MCTS_Node(len(self.actions_p1), RLforDL_MCTS_State(np.copy(test_input)))
        
        while not self.tc1(root.level, test_input, best_input, best_coverage):                
            if root.isLeaf() and self.tc3(root.level, test_input, root.state.mutated_input):
                if self.verbose:
                    print("Reached a game-over node")
                break
            iterations = 0
            while not self.tc2(iterations):
                current_node = root
                current_node.visit_count += 1

                # Selection until a leaf node
                while not current_node.isLeaf():
                    current_node = current_node.selection()
                    current_node.visit_count += 1
                    #print("selection", current_node.relative_index)
                
                # If not a terminating leaf
                if not self.tc3(current_node.level, test_input, current_node.state.mutated_input):
                    print("Expansion")
                    # Expansion (All children of the current leaf)
                    for i in range(len(current_node.child_nodes)):
                        if self.player_for_node(current_node) == 1:
                            nb_chidren_for_new_node = len(self.actions_p1)
                        else:
                            nb_chidren_for_new_node = len(self.actions_p2)
                        new_input = self.apply_action_for_node(current_node, i) 
                        new_node = MCTS_Node(nb_chidren_for_new_node, RLforDL_MCTS_State(new_input))
                        current_node.expansion(i, new_node)
                        
                    # Simulation
                    input_sim, coverage_sim, a1, a2 = self.simulate_for_node(current_node, test_input, coverage)
                    # Backpropagation
                    if self.player(current_node.level) == 1:
                        current_node = current_node.child_nodes[a1]
                    else:
                        current_node = current_node.child_nodes[a2]
                    current_node.backprop(coverage_sim)
                    print("current_node.backprop(coverage_sim)", coverage_sim, current_node.value, current_node.visit_count, current_node.level)

                    if coverage_sim > best_coverage:
                        best_input, best_coverage = input_sim, coverage_sim
                        if self.verbose_image:
                            plt.figure(2)
                            fig2.suptitle("BEST Coverage Increase: " + str(best_coverage))
                            for i in range(len(best_input[0:64])):
                                fig2_plots[i].set_data(best_input[i].reshape((28,28)))
                            fig2.canvas.flush_events()
                    
                if self.verbose:
                    print("Completed Iteration #%g" % (iterations))
                    print("Current Coverage From Simulation: %g" % (coverage_sim))
                    print("Best Coverage up to now: %g" % (best_coverage))
                iterations += 1
                    
            if self.verbose:
                print("Completed MCTS Level/Depth: #%g" % (root.level))
                root.printPath()
            
            try:
                root = root.bestChild(C)
            except Exception as e:
                print("except root", e)
                print("root.value", root.value)
                print("current_node.value", current_node.value)
                return root, best_input, best_coverage
        print("root.level", root.level)
        return root, best_input, best_coverage
