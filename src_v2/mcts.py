import numpy as np
import itertools

from src_v2.reward import Reward_Status

class MCTS_Node:
    def __init__(self, state, game, parent=None, relative_index=0):
        self.parent = parent
        self.relative_index = relative_index
        self.value = 0
        self.visit_count = 1
        self.child_nodes = [None] * state.nb_actions
        self.state = state
        self.game = game

    def potential(self, value=None, visit_count=None, parent_visit_count=None, C=np.sqrt(2)):
        # Upper Confidence Bound (UCB)
        # C: hyperparammer
        # See following for details on UCB:
        # 2008. G.M.J.B. Chaslot et all. Progressive strategies for monte-carlo tree search.
        # 2006. Levente Kocsis et all. Bandit based monte-carlo planning.
        if value == None and visit_count == None and parent_visit_count == None:
            value, visit_count, parent_visit_count = self.value, self.visit_count, self.parent.visit_count
        elif value == None or visit_count == None or parent_visit_count == None:
            raise Exception("set all value, visit_count and parent_visit_count parameters or leave all None")

        return value/visit_count + C*np.sqrt(np.log(parent_visit_count)/visit_count)

    def selection(self, C=np.sqrt(2)):
        p = []
        for child in self.child_nodes:
            if child != None:
                potential = child.potential(C=C)
            else:
                potential = self.potential(value=0,visit_count=1,parent_visit_count=self.visit_count, C=C)
            p.append(potential)

        p = np.array(p)

        if np.sum(p == 0) == p.size:
            p[:] = 1
        p /= p.sum()
        return np.random.choice(range(len(self.child_nodes)), p=p)

    def expansion(self, child_index):
        new_state, _ = self.game.step(self.state, child_index)
        if self.child_nodes[child_index] == None:
            self.child_nodes[child_index] = MCTS_Node(new_state, self.game, parent=self, relative_index=child_index)
        else:
            self.child_nodes[child_index].state = new_state
        return self.child_nodes[child_index]

    def simulation(self):
        current_node = self

        # take actions until a reward or game end
        while (not current_node.state.game_finished) and current_node.state.reward_status == Reward_Status.NOT_AVAILABLE:
            action = np.random.randint(0, current_node.state.nb_actions)
            current_node = current_node.expansion(action)
        
        return current_node, current_node.state.reward
    
    def backprop(self, reward):
        self.value += reward
        self.visit_count += 1
        self.state.visit()
        print("backprop", self, self.value, self.visit_count)
        if self.parent != None:
            self.parent.backprop(reward)

    def isLeaf(self):
        return self.child_nodes.count(None) == len(self.child_nodes)

    def bestChild(self, C=np.sqrt(2)):
        best = None
        best_r = 0

        for i in range(len(self.child_nodes)):
            if self.child_nodes[i] != None:
                print("bestChild", i, self.child_nodes[i], self.child_nodes[i].visit_count, self.child_nodes[i].value)
                if self.child_nodes[i].visit_count == 0:
                    current_r = 0
                else:
                    current_r = self.child_nodes[i].value / self.child_nodes[i].visit_count
                if best_r < current_r:
                    best = self.child_nodes[i]
                    best_r = current_r
        
        if best_r == 0:
            raise Exception("simulations failed to find any reward")
        else:
            return best

    def printPath(self, end="\n"):
        if self.parent:
            self.parent.printPath(end="->")
            print(self.relative_index, end=end)
        else:
            print("root", end=end)



def run_mcts(root, tc1, tc2, C=np.sqrt(2), verbose=True):
    while not tc1(root.state):                
        if root.isLeaf() and root.state.game_finished:
            if verbose:
                print("Reached a game-over node")
            break

        iterations = 0
        while not tc2(iterations):
            current_node = root

            # Selection until a leaf node
            selected_node_index = None
            while not current_node.isLeaf():
                node_index = current_node.selection()
                if current_node.child_nodes[node_index] != None:
                    current_node = current_node.child_nodes[node_index]
                else:
                    selected_node_index = node_index
                    break
                
            # Expansion
            if not current_node.state.game_finished:
                if selected_node_index == None:
                    selected_node_index = np.random.randint(0, current_node.state.nb_actions)
                current_node = current_node.expansion(selected_node_index)
            

            # If not a terminating leaf
            if not current_node.state.game_finished:
                # Simulation
                final_node, reward = current_node.simulation()
            
                # Backpropagation
                if reward != None:
                    final_node.backprop(reward)
            
            if verbose:
                print("Completed Iteration #%g" % (iterations))
                root.game.print_status()

            iterations += 1
                
        if verbose:
            print("Completed MCTS Level/Depth: #%g" % (root.state.level))
            root.printPath()
            root.game.print_status()
        
        try:
            root = root.bestChild(C)
        except Exception as e:
            print("Continuing with new batch:", e)
            return root

    return root
