import gym
from gym import error, spaces, utils
from gym.utils import seeding

from abc import ABC, abstractmethod

class DNN_Test_Base(ABC,gym.Env):
    def __init__(self, **kwargs):
        self._initialized = False
        self.observation_space = kwargs["observation_space"]
        self.action_space = kwargs["action_space"]
        if "input_chooser" in kwargs \
            and "input_extender" in kwargs \
                 and "coverage" in kwargs \
                     and "termination_criteria" in kwargs:
                     self.init_environment(kwargs["input_chooser"], kwargs["input_extender"],\
                          kwargs["coverage"], kwargs["termination_criteria"])

    def init_environment(self, input_chooser, input_extender, coverage, termination_criteria):
        self.input_chooser = input_chooser
        self.original_input, self.original_output = self.input_chooser()
        self.mutated_input, self.mutated_output = self.original_input, self.original_output
        self.input_extender = input_extender
        self.coverage = coverage
        self.termination_criteria = termination_criteria
        self.nb_steps = 0
        self._initialized = True
        print("SHAPE:", self.mutated_input.shape)

    def step(self, action):
        if not self._initialized:
            raise Exception("call init_environment() method first")
        else:
            self.nb_steps += 1
            self.apply_action(action)
            ob = self.mutated_input
            coverage_state, reward = self.coverage.step(ob, update_state=False)
            extended = self.input_extender(self.mutated_input, self.mutated_output, self.input_chooser, reward, self.original_input, self.original_output)
            if extended:
                self.coverage.step(ob, update_state=True, coverage_state=coverage_state)
            terminated = self.termination_criteria(self.original_input, self.mutated_input, self.original_output, self.mutated_output, self.nb_steps)
            return ob, 100*reward, terminated, {}

    def reset(self):
        if not self._initialized:
            raise Exception("call init_environment() method first")
        else:
            self.nb_steps = 0
            self.original_input, self.original_output = self.input_chooser()
            self.mutated_input, self.mutated_output = self.original_input, self.original_output
            ob = self.mutated_input
            return ob
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    @abstractmethod
    def apply_action(self, action):
        # applies the action specified in the param to self.mutated_input 
        pass