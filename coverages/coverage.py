from abc import ABC, abstractmethod
import numpy as np
import copy

def merge_state(old_state, new_state):
    if isinstance(old_state, list):
        final_state = []
        for i in range(len(old_state)):
            final_state.append(merge_state(old_state[i], new_state[i]))
    elif isinstance(old_state, dict):
        final_state = copy.deepcopy(old_state)
        for k in new_state.keys():
            if k in old_state:
                if np.isscalar(old_state[k]):
                    final_state[k] = new_state[k] if new_state[k]>old_state[k] else old_state[k] 
                elif isinstance(old_state[k], set):
                    final_state[k] = {*(old_state[k]), *(new_state[k])}
                else:
                    raise Exception("Unknown state subtype", type(old_state[k]))
            else:
                final_state[k] = new_state[k]
    elif isinstance(old_state, set):
        final_state = {*old_state, *new_state}
    else:
        raise Exception("Unknown state type")
    
    return final_state

coverage_call_count = 0

class AbstractCoverage(ABC):
    def step(self, test_inputs, update_state=True, coverage_state=None):
        global coverage_call_count
        coverage_call_count += 1
        print("coverage_call_count", coverage_call_count)
        old_state = copy.deepcopy(self.get_measure_state())
        old_coverage = self.get_current_coverage()
        self.reset_measure_state()
        if update_state:
            if coverage_state:
                self.set_measure_state(coverage_state)
            else:
                self.test(test_inputs)
                new_state = self.get_measure_state()
                final_state = merge_state(old_state, new_state)
                self.set_measure_state(final_state)
            new_coverage = self.get_current_coverage()
            return np.subtract(new_coverage, old_coverage)
        else:
            self.test(test_inputs)
            new_state = self.get_measure_state()
            final_state = merge_state(old_state, new_state)
            self.set_measure_state(final_state)
            new_coverage = self.get_current_coverage()
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
    def get_current_coverage(self):
        pass