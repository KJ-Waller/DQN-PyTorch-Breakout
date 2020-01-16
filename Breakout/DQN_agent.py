import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import time
from replay_mem import ReplayBuffer
from DQN_model import DQN
from transforms import Transforms
from PIL import Image

# This class trains and plays on the actual game
class DQAgent(object):
    # Take hyperparameters, as well as openai gym environment name
    # Keeps the environment in the class. All learning/playing functions are built in
    def __init__(self, replace_target_cnt, env, state_space, action_space, 
                model_name='breakout_model', gamma=0.99, eps_strt=0.1, 
                eps_end=0.001, eps_dec=5e-6, batch_size=32, lr=0.001):

        # Set global variables
        self.env = env
        self.state_space = state_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.GAMMA = gamma
        self.LR = lr
        self.eps = eps_strt
        self.eps_dec = eps_dec
        self.eps_end = eps_end

        # Use GPU if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialise Replay Memory
        self.memory = ReplayBuffer()

        # After how many training iterations the target network should update
        self.replace_target_cnt = replace_target_cnt
        self.learn_counter = 0

        # Initialise policy and target networks, set target network to eval mode
        self.policy_net = DQN(self.state_space, self.action_space, filename=model_name).to(self.device)
        self.target_net = DQN(self.state_space, self.action_space, filename=model_name+'target').to(self.device)
        self.target_net.eval()

        # If pretrained model of the modelname already exists, load it
        try:
            self.policy_net.load_model()
            print('loaded pretrained model')
        except:
            pass
        
        # Set target net to be the same as policy net
        self.replace_target_net()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss = torch.nn.SmoothL1Loss()

    def sample_batch(self):
        batch = self.memory.sample_batch(self.batch_size)
        state_shape = batch.state[0].shape

        # Convert to tensors with correct dimensions
        state = torch.tensor(batch.state).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).float().unsqueeze(1).to(self.device)
        state_ = torch.tensor(batch.state_).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        done = torch.tensor(batch.done).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    # Returns the greedy action according to the policy net
    def greedy_action(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        action = self.policy_net(obs).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, obs):
        if random.random() > self.eps:
            action = self.greedy_action(obs)
        else:
            action = random.choice([x for x in range(self.action_space)])
        return action
    
    # Stores a transition into memory
    def store_transition(self, *args):
        self.memory.add_transition(*args)

    # Updates the target net to have same weights as policy net
    def replace_target_net(self):
        if self.learn_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print('Target network replaced')

    # Decrement epsilon 
    def dec_eps(self):
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_end \
                        else self.eps_end

    # Samples a single batch according to batchsize and updates the policy net
    def learn(self, num_iters=1):
        if self.memory.pointer < self.batch_size:
            return 

        for i in range(num_iters):

            # Sample batch
            state, action, reward, state_, done = self.sample_batch()

            # Calculate the value of the action taken
            q_eval = self.policy_net(state).gather(1, action)

            # Calculate best next action value from the target net and detach from graph
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze(1)
            # Using q_next and reward, calculate q_target
            # (1-done) ensures q_target is 0 if transition is in a terminating state
            q_target = (1-done) * (reward + self.GAMMA * q_next) + (done * reward)

            # Compute the loss
            # loss = self.loss(q_target, q_eval).to(self.device)
            loss = self.loss(q_eval, q_target).to(self.device)

            # Perform backward propagation and optimization step
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Increment learn_counter (for dec_eps and replace_target_net)
            self.learn_counter += 1

            # Check replace target net
            self.replace_target_net()

        # Save model & decrement epsilon
        self.policy_net.save_model()
        self.dec_eps()

    # Save gif of an episode starting num_transitions ago from memory
    def save_gif(self, num_transitions):
        frames = []
        for i in range(self.memory.pointer - num_transitions, self.memory.pointer):
            frame = Image.fromarray(self.memory.memory[i].raw_state, mode='RGB')
            frames.append(frame)
        
        frames[0].save('episode.gif', format='GIF', append_images=frames[1:], save_all=True, duration=10, loop=0)

    # Plays num_eps amount of games, while optimizing the model after each episode
    def train(self, num_eps=100, render=False):
        scores = []

        max_score = 0

        for i in range(num_eps):
            done = False

            # Reset environment and preprocess state
            obs = self.env.reset()
            state = Transforms.to_gray(obs)
            
            score = 0
            cnt = 0
            while not done:
                # Take epsilon greedy action
                action = self.choose_action(state)
                obs_, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()

                # Preprocess next state and store transition
                state_ = Transforms.to_gray(obs, obs_)
                self.store_transition(state, action, reward, state_, int(done), obs)

                score += reward
                obs = obs_
                state = state_
                cnt += 1

            # Maintain record of the max score achieved so far
            if score > max_score:
                max_score = score

            # Save a gif if episode is best so far
            if score > 300 and score >= max_score:
                self.save_gif(cnt)

            scores.append(score)
            print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
                \n\tEpsilon: {self.eps}\n\tTransitions added: {cnt}')
            
            # Train on as many transitions as there have been added in the episode
            print(f'Learning x{math.ceil(cnt/self.batch_size)}')
            self.learn(math.ceil(cnt/self.batch_size))

        self.env.close()

    # This function simply lets a pretrained model be evaluated to play a game
    # No learning will be done
    def play_games(self, num_eps, render=True):

        # Set network to eval mode
        self.policy_net.eval()

        scores = []

        for i in range(num_eps):
            done = False

            # Get observation and preprocess
            obs = self.env.reset()
            state = Transforms.to_gray(obs)
            
            score = 0
            cnt = 0
            while not done:
                # Take the greedy action and observe next state
                action = self.greedy_action(state)
                obs_, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()

                # Preprocess next state and store transition
                state_ = Transforms.to_gray(obs, obs_)
                self.store_transition(state, action, reward, state_, int(done), obs)

                # Calculate score, set next state and obs and increment counter
                score += reward
                obs = obs_
                state = state_
                cnt += 1

            # If the score is more than 300, save a gif of that game
            if score > 300:
                self.save_gif(cnt)
            
            scores.append(score)
            print(f'Episode {i}/{num_eps}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
                \n\tEpsilon: {self.eps}\n\tSteps made: {cnt}')

        
        self.env.close()