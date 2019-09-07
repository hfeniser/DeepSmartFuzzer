
import itertools
import numpy as np
import src.image_transforms as image_transforms
import copy

from src_v2.utility import find_the_distance, init_image_plots
from src_v2.reward import Reward_Status

import pprint

pp = pprint.PrettyPrinter()

class RLforDL_State:
    def __init__(self, mutated_input, action=0, previous_state=None, reward_status=Reward_Status.NOT_AVAILABLE, reward=None, game=None):
        self.mutated_input = copy.deepcopy(mutated_input)

        self.previous_state = previous_state
        if self.previous_state != None:
            self.original_input = copy.deepcopy(previous_state.original_input)
            self.level = previous_state.level+1
            self.action_history = previous_state.action_history + [action]
            self.game = previous_state.game  
        else:
            self.original_input = copy.deepcopy(mutated_input)
            self.level = 0
            self.action_history = []
            self.game = game

        self.nb_actions = self.game.get_nb_actions(self.level)
        self.game_finished = False
        self.reward_status = reward_status
        if self.reward_status != Reward_Status.NOT_AVAILABLE:
            self.input_changed = True
        else:
            self.input_changed = False
        self.reward = reward
    
    def visit(self):
        if self.reward_status == Reward_Status.UNVISITED:
            self.reward_status = Reward_Status.VISITED

class RLforDL:
    def __init__(self, coverage, input_shape, input_lower_limit, input_upper_limit, action_division_p1, actions_p2, ending_condition, with_implicit_reward=False, distance_in_reward=False, verbose=True, verbose_image=True):
        self.coverage = coverage
        self.input_shape = input_shape
        self.input_lower_limit = input_lower_limit
        self.input_upper_limit = input_upper_limit
        options_p1 = []
        self.actions_p1_spacing = []
        for i in range(len(action_division_p1)):
            spacing = int(self.input_shape[i] / action_division_p1[i])
            if self.input_shape[i] == 1:
                options_p1.append([0])
            else:
                options_p1.append(list(range(0, self.input_shape[i]-spacing+1, spacing)))
            self.actions_p1_spacing.append(spacing)

        actions_p1_lower_limit = np.array(list(itertools.product(*options_p1)))
        actions_p1_upper_limit = np.add(actions_p1_lower_limit, self.actions_p1_spacing)

        # round upper_limits of end/edge section to input_shape (NO EXACT DIVISION)
        for i in range(len(action_division_p1)):
            if self.input_shape[i] != 1:
               round_up = actions_p1_upper_limit[:,i] > (self.input_shape[i] - self.actions_p1_spacing[i])
               actions_p1_upper_limit[:,i] = round_up * self.input_shape[i] + np.logical_not(round_up) * actions_p1_upper_limit[:,i]
        
        self.actions_p1 = []
        for i in range(len(actions_p1_lower_limit)):
            self.actions_p1.append({
                "lower_limits": actions_p1_lower_limit[i],
                "upper_limits": actions_p1_upper_limit[i]
            })
        
        self.actions_p2 = actions_p2
        self.ending_condition = ending_condition
        self.with_implicit_reward = with_implicit_reward
        self.distance_in_reward = distance_in_reward
        self.verbose = verbose
        self.verbose_image = verbose_image

        self.best_reward = 0
        self.best_input = None

        if self.verbose:
            print("self.actions_p1") 
            pp.pprint(self.actions_p1)
            print("self.actions_p2")
            pp.pprint(self.actions_p2)

        if self.verbose_image:
            self.fig_current, self.fig_plots_current = init_image_plots(8, 8, self.input_shape[1:3])
            self.fig_best, self.fig_plots_best = init_image_plots(8, 8, self.input_shape[1:3]) 
    
    def get_stat(self):
        return self.best_reward, self.best_input

    def reset_stat(self):
        self.best_reward = 0
        self.best_input = None

    def player(self, level):
        if level % 2 == 0:
            return 1
        else:
            return 2
    
    def get_nb_actions(self, level):
        if self.player(level) == 1:
            return len(self.actions_p1)
        else:
            return len(self.actions_p2)

    def apply_action(self, state, action1, action2):
        mutated_input = copy.deepcopy(state.mutated_input)
        
        # get action details
        action_part1 = self.actions_p1[action1]
        action_part2 = self.actions_p2[action2]
        
        # find the part of input where the mutation will be applied
        lower_limits = action_part1['lower_limits']
        upper_limits = action_part1['upper_limits']
        s = tuple([slice(lower_limits[i], upper_limits[i]) for i in range(len(lower_limits))])
        
        # loop through the inputs and apply the mutation on each
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

    def calc_reward(self, mutated_input):
        _, reward = self.coverage.step(mutated_input, update_state=False, with_implicit_reward=self.with_implicit_reward)
        #reward = self.reward_shaping()

        return reward

    # def reward_shaping(self, mutated_input, original_input, reward):
        # if self.distance_in_reward:
        #     dist = find_the_distance(input_sim, node)
        #     if reward > 0 and dist > 0:
        #         reward = reward / dist
    
    def step(self, state, action, return_reward=True):
        if self.player(state.level) == 1:
            new_state = RLforDL_State(state.mutated_input, action=action, previous_state=state, reward_status=Reward_Status.NOT_AVAILABLE)
        else:
            action1 = state.action_history[-1]
            action2 = action
            mutated_input = self.apply_action(state, action1, action2)
            reward = self.calc_reward(mutated_input)
            new_state = RLforDL_State(mutated_input, action=action, previous_state=state, reward_status=Reward_Status.UNVISITED, reward=reward)
        
        if self.ending_condition(new_state):
            # already an termination node
            new_state.game_finished = True

        if new_state.reward != None:
            if self.verbose:
                print("Reward:", new_state.reward)
            
            if self.verbose_image:
                self.fig_current.suptitle("level:" + str(new_state.level) + " Action: " + str((action1,action2)) + " Reward: " + str(new_state.reward))
                for i in range(len(new_state.mutated_input[0:64])):
                    self.fig_plots_current[i].set_data(new_state.mutated_input[i].reshape(self.input_shape[1:3]))
                self.fig_current.canvas.flush_events()

            if new_state.reward > self.best_reward:
                self.best_input, self.best_reward = copy.deepcopy(new_state.mutated_input), new_state.reward

                if self.verbose_image:
                    self.fig_best.suptitle("Best Reward: " + str(self.best_reward))
                    for i in range(len(self.best_input[0:64])):
                        self.fig_plots_best[i].set_data(self.best_input[i].reshape(self.input_shape[1:3]))
                    self.fig_best.canvas.flush_events()

        return new_state, new_state.reward

    def print_status(self):
        print("Best Reward:", self.best_reward)
