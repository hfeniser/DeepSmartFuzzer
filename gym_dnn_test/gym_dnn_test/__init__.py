from gym.envs.registration import register
from gym import spaces
import numpy as np

mnist_ob_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(1, 28, 28, 1))
mnist_action_space = spaces.Tuple((spaces.Discrete(1), 
                                   spaces.Discrete(28), 
                                   spaces.Discrete(28), 
                                   spaces.Discrete(1), 
                                   spaces.Box(low=-255, high=255, dtype=np.int16, shape=(1,))))
register(
    id='mnist-v0',
    entry_point='gym_dnn_test.envs:DNN_Test_V0',
    kwargs={
        'observation_space': mnist_ob_space,
        'action_space': mnist_action_space
    }
)