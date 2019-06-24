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
        return self.value + C*np.sqrt(np.log(self.parent.visit_count)/self.visit_count)

    def selection(self, C=np.sqrt(2)):
        p = [child.potential(C) for child in self.child_nodes]
        return np.random.choice(range(len(self.child_nodes)), p=p)

    def expansion(self, child_index, new_node):
        new_node.parent = self
        new_node.level = self.level+1
        new_node.relative_index = child_index
        self.child_nodes[child_index] = new_node

    def backprop(self, reward):
        self.value += reward
        if self.parent:
            self.parent.backprop(reward)

    def isLeaf(self):
        return self.child_nodes.count(None) == len(self.child_nodes)

    def bestChild(self, C=np.sqrt(2)):
        best = self.child_nodes[0]
        for i in range(1, len(self.child_nodes)):
            if best.potential(C) < self.child_nodes[i].potential(C):
                best = self.child_nodes[i]
        
        return best


class RLforDL_MCTS_State:
    def __init__(self, mutated_input):
        self.mutated_input = mutated_input

class RLforDL_MCTS:
    def __init__(self, input_shape, action_division_p1, actions_p2, tc1, tc2, tc3):
        self.input_shape = input_shape
        options_p1 = []
        self.actions_p1_spacing = []
        for i in range(len(action_division_p1)):
            spacing = int(self.input_shape[i] / action_division_p1[i])
            options_p1.push(list(range(0, self.input_shape[i], spacing)))
            self.actions_p1_spacing.push(spacing)

        self.actions_p1 = list(itertools.product(*options_p1))
        self.actions_p2 = actions_p2
        self.tc1 = tc1 # termination condition for the entire program e.g. limit the number of epochs
        self.tc2 = tc2 # termination condition for the current iteration e.g. limit the iterations
        self.tc3 = tc3 # cut-off condition for the tree
    
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
        lower_limits = np.add(action_part1, actions_p1_spacing)
        upper_limits = np.subtract(action_part1, actions_p1_spacing)
        s = tuple([slice(lower_limits[i], upper_limits[i]) for i in range(len(lower_limits))])
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
        input_sim = node.state.mutated_input

        if self.player(level) == 1:
            action1 = np.random.randint(0,len(self.actions_p1))
        else:
            action2 = np.random.randint(0,len(self.actions_p2))
            input_sim = self.apply_action_for_node(node, action2)

        _, coverage_sim = coverage.step(input_sim, update_state=False)
    
        while not self.tc3(level, test_input, input_sim):
            level += 1
            if self.player(level) == 1:
                action1 = np.random.randint(0,len(self.actions_p1))
            else:
                action2 = np.random.randint(0,len(self.actions_p2))
                input_sim = self.apply_action(input_sim, action1, action2)
                _, coverage_sim = coverage.step(input_sim, update_state=False)
        
        return input_sim, coverage_sim

    def run(self, test_input, coverage, C=np.sqrt(2)):
        best_input, best_coverage = None, 0
        root = MCTS_Node(len(self.actions_p1), RLforDL_MCTS_State(np.copy(test_input)))
        while not self.tc1(root, test_input, best_input, best_coverage):
            while not self.tc2(root):
                # start a new iteration from root
                current_node = root
                current_node.visit_count += 1

                # Selection until a leaf node
                while not current_node.isLeaf():
                    current_node = current_node.selection()
                    current_node.visit_count += 1
                
                # If not a terminating leaf
                if not self.tc3(current_node.level, test_input, current_node.state.mutated_input):
                    # Expansion (All children of the current leaf)
                    for i in range(len(current_node.child_nodes)):
                        if self.player_for_node(current_node) == 1:
                            nb_chidren_for_new_node = len(self.actions_p1)
                        else:
                            nb_chidren_for_new_node = len(self.actions_p2)
                        new_input = self.apply_action_for_node(current_node, i) 
                        new_node = MCTS_Node(nb_chidren_for_new_node, RLforDL_MCTS_State(new_input))
                        current_node.expansion(i, new_node)

                    # Simulation & Backpropogation (All children of the current leaf)
                    for node in current_node.child_nodes:
                        input_sim, coverage_sim = self.simulate_for_node(node, test_input, coverage)
                        node.backprop(coverage_sim)
                        if coverage_sim > best_coverage:
                            best_input, best_coverage = input_sim, coverage_sim


            root = root.bestChild(C)
        
        return root, best_input, best_coverage
