from abc import ABC, abstractmethod
import numpy as np

class AbstractCoverage(ABC):
    def __init__(self, model, state_len_calc_input):
        super().__init__()
        self.model = model
        self.state_len = self._calc_state_len(state_len_calc_input)
        self.state = np.zeros(self.state_len)
        self.coverage = 0

    def get_len_of_state(self):
        return self.state_len 
    
    def get_current_state(self):
        return self.state
    
    def get_current_coverage(self):
        return self.coverage
    
    def step(self, test_inputs, update_state=True, coverage_state=None, *argv, **kwargs):
        # calculates new coverage using test_input, 
        # sets self.state, self.coverage accordingly
        # and returns reward (new coverage - old coverage)
        if coverage_state is None:
            coverage_state = self._calc_coverage(test_inputs, *argv, **kwargs)
        
        new_state = np.maximum(coverage_state, self.state)
        new_coverage = np.sum(new_state) / self.state_len
        reward = new_coverage - self.coverage
        if update_state:
            self.state = new_state
            self.coverage = new_coverage
            return reward
        else:
            return coverage_state, reward
    
    @abstractmethod
    def _calc_state_len(self, test_inputs, *argv, **kwargs):
        # calculates coverage state length using state_len_calc_input
        pass
    
    @abstractmethod
    def _calc_coverage(self, test_inputs, *argv, **kwargs):
        # calculates new coverage using test_input and returns coverage and coverage state
        pass
    
from neuron_cov import measure_neuron_cov

class NeuronCoverage(AbstractCoverage):
    def _calc_state_len(self, test_inputs, *argv, **kwargs):
        # calculates coverage state length using state_len_calc_input
        _, _, total, _, _  = measure_neuron_cov(self.model, test_inputs, *argv, **kwargs)
        return total
   
    def _calc_coverage(self, test_inputs, *argv, **kwargs):
        # calculates new coverage using test_input and returns coverage and coverage state
        _, _, _, _, coverage_state = measure_neuron_cov(self.model, test_inputs, *argv, **kwargs)
        return coverage_state

    
    
from kmn import measure_k_multisection_cov
from utils import calc_major_func_regions 

class KMultisectionCoverage(AbstractCoverage):
    def __init__(self, model, state_len_calc_input, k, train_inputs, skip=[]):
        self.k = k
        self.skip = skip
        self.major_func_regions = calc_major_func_regions(model, train_inputs, self.skip)
        super().__init__(model, state_len_calc_input)
        
    def _calc_state_len(self, test_inputs, *argv, **kwargs):
        # calculates coverage state length using state_len_calc_input
        return len(self._calc_coverage(test_inputs, *argv, **kwargs))
   
    def _calc_coverage(self, test_inputs, *argv, **kwargs):
        # calculates new coverage using test_input and returns coverage and coverage state
        response = measure_k_multisection_cov(self.model, test_inputs, self.k, *argv, major_func_regions=self.major_func_regions, skip=self.skip, **kwargs)
        # response:
        #(percent_str(multisection_activated, k*total), multisection_activated,
        #        percent_str(upper_activated+lower_activated, 2 * total),
        #        percent_str(upper_activated, total),
        #        lower_activated, upper_activated, total,
        #        multisection_activated, upper_activated, lower_activated, total, outs,
        #        multisection_coverage_state, upperlower_coverage_state, upper_coverage_state)
        return response[-3] #multisection_coverage_state