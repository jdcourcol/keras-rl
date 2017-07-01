''' create race agents '''
import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, merge
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory
from rl.agents import SARSAAgent, DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
def create_environment(log_file_name=None):
    ''' create gym env '''
    ENV_NAME = 'F1-v0'
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    return env


def create_deep_model(env):
    ''' create a deep NN model '''
    nb_actions = env.action_space.n
    model = Sequential()
    model.add(Flatten(input_shape=(1, ) + (env.observation_space.shape[1], )))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


def create_dqn_agent(env):
    ''' create dqn agent '''
    model = create_deep_model(env)
    nb_actions = env.action_space.n
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=500,
        target_model_update=1e-2,
        policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn


def create_cem_agent(env):
    ''' create cem agent '''
    env = create_environment()
    model = create_deep_model(env)
    nb_actions = env.action_space.n
    memory = EpisodeParameterMemory(limit=1000, window_length=1)
    cem = CEMAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        batch_size=50,
        nb_steps_warmup=2000,
        train_interval=50,
        elite_frac=0.05)
    cem.compile()
    return cem

def create_sarsa_agent(env):
    env = create_environment()
    model = create_deep_model(env)
    nb_actions = env.action_space.n
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
    sarsa.compile(Adam(lr=1e-3), metrics=['mae'])
    return sarsa

def create_ddpg_agent(env):
    
    nb_actions = env.action_space.n
    policy = BoltzmannQPolicy()
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + (env.observation_space.shape[1],)))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + (env.observation_space.shape[1],), name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    return agent

def fit_agent(env, agent, filename):
    ''' fit agent '''
    agent.fit(env, nb_steps=250000, visualize=True, verbose=2)
    # After training is done, we save the final weights.
    agent.save_weights(filename, overwrite=True)


def test_agent(env, agent):
    ''' test agent '''
    agent.test(env, nb_episodes=5, visualize=True)

POSSIBLE_AGENTS = {'ddpg': create_ddpg_agent}
# {'dqn': create_dqn_agent}
# , 'cem': create_cem_agent}
