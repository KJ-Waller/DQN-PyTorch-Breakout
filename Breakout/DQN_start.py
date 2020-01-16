from DQN_agent import DQAgent
import gym
from transforms import Transforms
import numpy as np

# Specify environment location
env_name = 'Breakout-v0'

# Initializes an openai gym environment
def init_gym_env(env_path):

    env = gym.make(env_path)

    state_space = env.reset().shape
    state_space = (state_space[2], state_space[0], state_space[1])
    state_raw = np.zeros(state_space, dtype=np.uint8)
    processed_state = Transforms.to_gray(state_raw)
    state_space = processed_state.shape
    action_space = env.action_space.n

    return env, state_space, action_space

# Initialize Gym Environment
env, state_space, action_space = init_gym_env(env_name)
    
# Create an agent
agent = DQAgent(replace_target_cnt=5000, env=env, state_space=state_space, action_space=action_space, model_name='breakout_model', gamma=.99,
                eps_strt=.1, eps_end=.001, eps_dec=5e-6, batch_size=32, lr=.001)

# Train num_eps amount of times and save onnx model
agent.train(num_eps=75000)