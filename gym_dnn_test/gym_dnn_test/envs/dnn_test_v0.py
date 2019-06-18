from gym_dnn_test.envs.dnn_test_base import DNN_Test_Base

class DNN_Test_V0(DNN_Test_Base):
    def apply_action(self, action):
        self.mutated_input[tuple(action[:-1])] += action[-1]