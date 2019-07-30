from abc import ABC, abstractmethod
import numpy as np
import copy


coverage_call_count = 0

class AbstractCoverage(ABC):
    def step(self, test_inputs, update_state=True, coverage_state=None, with_implicit_reward=False):
        global coverage_call_count
        coverage_call_count += 1
        if coverage_call_count % 100 == 0:
            print("coverage_call_count", coverage_call_count)
        old_state = copy.deepcopy(self.get_measure_state())
        old_coverage = self.get_current_coverage()
        if update_state:
            if coverage_state:
                self.set_measure_state(coverage_state)
            else:
                self.test(test_inputs)
            new_coverage = self.get_current_coverage(with_implicit_reward=with_implicit_reward)
            return np.subtract(new_coverage, old_coverage)
        else:
            self.test(test_inputs)
            new_state = self.get_measure_state()
            new_coverage = self.get_current_coverage(with_implicit_reward=with_implicit_reward)
            self.set_measure_state(old_state)
            return new_state, np.subtract(new_coverage, old_coverage)
    
    @abstractmethod
    def get_measure_state(self):
        pass

    @abstractmethod
    def set_measure_state(self, state):
        pass

    @abstractmethod
    def reset_measure_state(self):
        pass
    
    @abstractmethod
    def reset_implicit_reward_state(self):
        pass 

    @abstractmethod
    def get_current_coverage(self, with_implicit_reward=False):
        pass