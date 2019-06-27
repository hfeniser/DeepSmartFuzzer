import numpy as np
import itertools

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
        p /= p.sum()
        return self.child_nodes[np.random.choice(range(len(self.child_nodes)), p=p)]

    def expansion(self, child_index, new_node):
        new_node.parent = self
        new_node.level = self.level+1
        new_node.relative_index = child_index
        self.child_nodes[child_index] = new_node

    def backprop(self, reward):
        if reward != 0:
            self.value += reward
            if self.parent:
                self.parent.backprop(reward)

    def isLeaf(self):
        return self.child_nodes.count(None) == len(self.child_nodes)

    def bestChild(self, C=np.sqrt(2)):
        best = self.child_nodes[0]
        best_r = best.value / best.visit_count

        for i in range(1, len(self.child_nodes)):
            current_r = self.child_nodes[i].value / self.child_nodes[i].visit_count
            if best_r < current_r:
                best = self.child_nodes[i]
                best_r = current_r

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

class RLforDL_MCTS:
    def __init__(self, input_shape, action_division_p1, actions_p2, tc1, tc2, tc3, verbose=True):
        self.input_shape = input_shape
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

        self.verbose = verbose
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
        lower_limits = np.subtract(action_part1, self.actions_p1_spacing)
        lower_limits = (lower_limits >= 0)*lower_limits
        upper_limits = np.add(action_part1, self.actions_p1_spacing)
        upper_limits = (upper_limits <= self.input_shape)*upper_limits \
                        + (upper_limits > self.input_shape)*self.input_shape
        s = tuple([slice(lower_limits[i], upper_limits[i]) for i in range(len(lower_limits))])
        #if self.verbose:
        #    print("action", action1, action2)
        mutated_input[s] += action_part2

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
        best_input_sim, best_coverage_sim = np.copy(input_sim), 0

        if self.player(level) == 1:
            action1 = np.random.randint(0,len(self.actions_p1))
        else:
            action2 = np.random.randint(0,len(self.actions_p2))
            input_sim = self.apply_action_for_node(node, action2)
        
        input_changed = True

        while not self.tc3(level, test_input, input_sim):
            if input_changed:
                _, coverage_sim = coverage.step(input_sim, update_state=False)
                if coverage_sim > best_coverage_sim:
                    best_input_sim, best_coverage_sim = np.copy(input_sim), coverage_sim
                input_changed = False
            
            level += 1
            if self.player(level) == 1:
                action1 = np.random.randint(0,len(self.actions_p1))
            else:
                action2 = np.random.randint(0,len(self.actions_p2))
                input_sim = self.apply_action(input_sim, action1, action2)
                input_changed = True

        return best_input_sim, best_coverage_sim

    def run(self, test_input, coverage, C=np.sqrt(2)):
        best_input, best_coverage = test_input, 0
        root = MCTS_Node(len(self.actions_p1), RLforDL_MCTS_State(np.copy(test_input)))
        while not self.tc1(root.level, test_input, best_input, best_coverage):                
            if root.isLeaf() and self.tc3(root.level, test_input, root.state.mutated_input):
                if self.verbose:
                    print("Reached a game-over node")
                break
            
            iterations = 0
            while not self.tc2(iterations):
                # start a new iteration from root
                current_node = root
                current_node.visit_count += 1

                # Selection until a leaf node
                while not current_node.isLeaf():
                    current_node = current_node.selection()
                    current_node.visit_count += 1
                    #print("selection", current_node.relative_index)
                
                # If not a terminating leaf
                if not self.tc3(current_node.level, test_input, current_node.state.mutated_input):
                    # Expansion (All children of the current leaf)
                    for i in range(len(current_node.child_nodes)):
                        if self.player_for_node(current_node) == 1:
                            nb_chidren_for_new_node = len(self.actions_p2)
                        else:
                            nb_chidren_for_new_node = len(self.actions_p1)
                        new_input = self.apply_action_for_node(current_node, i) 
                        new_node = MCTS_Node(nb_chidren_for_new_node, RLforDL_MCTS_State(new_input))
                        current_node.expansion(i, new_node)
                    
                    # Simulation & Backpropogation (All children of the current leaf)
                    for node in current_node.child_nodes:
                        input_sim, coverage_sim = self.simulate_for_node(node, test_input, coverage)
                        node.backprop(coverage_sim)
                        if coverage_sim > best_coverage:
                            best_input, best_coverage = input_sim, coverage_sim
                
                if self.verbose:
                    print("Completed Iteration #%g" % (iterations))
                    print("Current Coverage From Simulation: %g" % (coverage_sim))
                    print("Best Coverage up to now: %g" % (best_coverage))
                iterations += 1
            
            if self.verbose:
                print("Completed MCTS Level/Depth: #%g" % (root.level))
                root.printPath()
            
            root = root.bestChild(C)
        
        return root, best_input, best_coverage
