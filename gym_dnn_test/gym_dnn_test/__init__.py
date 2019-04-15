from gym.envs.registration import register
from gym import spaces
import numpy as np


def register_dnn_test_env_from_config(id, config, entry_point='gym_dnn_test.envs:DNN_Test_V0'):
    register(
        id=id,
        entry_point=entry_point,
        kwargs=config
    )