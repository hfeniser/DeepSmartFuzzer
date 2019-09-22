from abc import ABC, abstractmethod
import numpy as np
import copy


coverage_call_count = 0

class AbstractCoverage(ABC):
    def step(self, test_inputs, update_state=True, coverage_state=None, with_implicit_reward=False):
        global coverage_call_count
        print("COVERAGE CALL!!!!!!!!!!!!")
        coverage_call_count += 1
        if coverage_call_count % 100 == 0:
            print("coverage_call_count", coverage_call_count)
        old_state = copy.deepcopy(self.get_measure_state())
        old_coverage = self.get_current_coverage(with_implicit_reward=False)
        if update_state:
            if coverage_state:
                self.set_measure_state(coverage_state)
            else:
                self.test(test_inputs, with_implicit_reward=with_implicit_reward)
            new_coverage = self.get_current_coverage(with_implicit_reward=with_implicit_reward)
            return np.subtract(new_coverage, old_coverage)
        else:
            self.test(test_inputs, with_implicit_reward=with_implicit_reward)
            new_state = self.get_measure_state()
            new_coverage = self.get_current_coverage(with_implicit_reward=with_implicit_reward)
            self.set_measure_state(old_state)
            #print("new_coverage, old_coverage, s", new_coverage, old_coverage, np.subtract(new_coverage, old_coverage))
            return new_state, np.subtract(new_coverage, old_coverage)

    def calc_reward(self, activation_table, with_implicit_reward=False):
        activation_values = np.array(list(activation_table.values()))
        #print("activation_values", activation_values)
        covered_positions = activation_values == 1
        covered = np.sum(covered_positions)
        if with_implicit_reward and self.calc_implicit_reward:
            implicit_reward = self.calc_implicit_reward(activation_values, covered_positions)
        else:
            implicit_reward = 0
        reward = covered + implicit_reward
        #print("reward, covered, implicit_reward", reward, covered, implicit_reward)
        return reward, covered, implicit_reward
    
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
    def get_current_coverage(self, with_implicit_reward=False):
        pass

    @abstractmethod
    def test(self, test_inputs, with_implicit_reward=False):
        pass