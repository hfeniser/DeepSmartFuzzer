import argparse

parser = argparse.ArgumentParser(description='Script for testing LeNet models for MNIST dataset by Reinforcement Learning')
parser.add_argument("--lenet", type=int, default=1, choices=[1,4,5])
parser.add_argument("--rl-algorithm", type=str, default="dqn", choices=["dqn"])
args = parser.parse_args()

print("Arguments:", args)

from models.lenet_models import LeNet1, LeNet4, LeNet5
from keras.layers import Input
from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
import gym
import os

from input_chooser import InputChooser
from coverages.coverage import KMultisectionCoverage

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(-1,28,28,1).astype(np.int16)
test_images = test_images.reshape(-1,28,28,1).astype(np.int16)


#multiple instance of openmp error 
#solution: set environment variable
#KMP_DUPLICATE_LIB_OK=TRUE
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if args.lenet == 1:
    model = LeNet1(Input((28,28,1)))
elif args.lenet == 4:
    model = LeNet4(Input((28,28,1)))
elif args.lenet == 5:
    model = LeNet5(Input((28,28,1)))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

input_chooser = InputChooser(test_images, test_labels)

(state_len_calc_input, k, train_inputs) = (test_images[0].reshape(-1, 28, 28, 1), 10000, train_images.reshape(-1, 28, 28, 1))
coverage = KMultisectionCoverage(model, state_len_calc_input, k, train_inputs)
initial_coverage = coverage.step(test_images.reshape(-1,28,28,1))
print("initial coverage:  %g" % (initial_coverage))

def input_extender(mutated_input, mutated_output, input_chooser, reward, original_input, original_output): 
    if reward >= 0.0018:
        input_chooser.append(mutated_input, mutated_output)
        return True
    else:
        return False

def termination_criteria(original_input, mutated_input, original_output, mutated_output, nb_steps):
    return nb_steps >= 30 or np.max(mutated_input) > 255 or np.min(mutated_input) < 0 


from gym_dnn_test import register_dnn_test_env_from_config
from gym import spaces

mnist_ob_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(1, 28, 28, 1))
mnist_action_space = spaces.Tuple((spaces.Discrete(1), 
                                   spaces.Discrete(28), 
                                   spaces.Discrete(28), 
                                   spaces.Discrete(1), 
                                   spaces.Box(low=-255, high=255, dtype=np.int16, shape=(1,))))

env_config = {
    "observation_space": mnist_ob_space,
    "action_space": mnist_action_space,
    "input_chooser": input_chooser,
    "input_extender": input_extender,
    "coverage": coverage,
    "termination_criteria": termination_criteria
} 
env_id = 'mnist-v1'
register_dnn_test_env_from_config(env_id, env_config)
env = gym.make('gym_dnn_test:mnist-v1')

print("environment initialized")

from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


def WrapMNISTEnv(env):
    def reshape_action(action):
        a = tuple()
        a += (action%(255*2 + 1)-255, )
        action = int(action/(255*2 + 1))
        a += (0, )
        a += (action%28, )
        action = int(action/28)
        a += (action%28, )
        a += (0, )
        return tuple(reversed(a))
    _step = env.step
    env.step = lambda action: _step(reshape_action(action))
    return env

env = WrapMNISTEnv(env)

if args.rl_algorithm == "dqn":
    # derived from https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
    nb_actions = 1*28*28*1*(255*2 + 1)

    # Model (similar to lenet1, only last layer is replaced)
    input_tensor = Input((1,1,28,28,1))
    x = Reshape((28,28,1))(input_tensor)
    # block1
    x = Convolution2D(4, (5, 5), activation='relu', padding='same', name='block1_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(12, (5, 5), activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='dense1')(x)
    x = Dense(nb_actions, activation='linear', name='output')(x)
    model = Model(input_tensor, x)
    print(model.summary())

    memory = SequentialMemory(limit=1000000, window_length=1)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)
    
    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                nb_steps_warmup=5000, gamma=.99, target_model_update=1000,
                train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])


    experiment_folder = 'experiments/dqn_mnist_lenet'

    os.makedirs(experiment_folder, exist_ok=True)

    weights_filename = experiment_folder + '/weights.h5f'
    checkpoint_weights_filename = experiment_folder + '/checkpoint_weights_{step}.h5f'
    log_filename = experiment_folder + '/log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=25000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=200000, log_interval=1000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    np.save(experiment_folder + '/extended_testset_inputs.npy', input_chooser.features)
    np.save(experiment_folder + '/extended_testset_outputs.npy', input_chooser.labels)

elif args.rl_algorithm == "ddpg":
    pass
elif args.rl_algorithm == "sac":
    pass