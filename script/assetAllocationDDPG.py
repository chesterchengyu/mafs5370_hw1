import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import rl_utils
from ddpg import DDPG

# register environment

from gym.envs.registration import register

register(
    id='AssetAllocationEnv-v0',
    entry_point='env.Assetallocate:AssetAllocationEnv',
    max_episode_steps=1000,
)


class assetAllocationDDPG:

    '''
    This class contains functions:
    - running asset allocation by using DDPG
    - computing Q value
    - getting final optimal strategy and Q value
    - plotting moving average of the final utility across different training episodes
    --------------------------------------------------------------------------------------------------------------------
    11/2/2023 | LAI Fujie | Initial specification of the algorithm
    25/2/2023 | CHENG Yu | Refactor and put everything into the class
    4/3/2023 | CHENG Yu | Add function to output the average of final several rounds of episode as output
    '''

    def __init__(self, initial_wealth, p, a, b, risk_free, T, env_id='AssetAllocationEnv-v0',
                 actor_lr=3e-4, critic_lr=3e-3, num_episodes=1000, hidden_dim=128, gamma=0.98,
                 tau=0.005, buffer_size=10000, minimal_size=1000, batch_size=128, sigma=0.05, epsilon=0.01):
        '''
        asset allocation by DDPG
        :param initial_wealth: float - initial wealth
        :param p: float - probability of risky asset to go up after 1 period
        :param a: float - ratio between the discounted value of risk asset will become after 1 period if it goes up and initial value
        :param b: float - ratio between the discounted value of risky asset will become after 1 period it goes down and initial value
        :param risk_free: float - value of discounted risk-free asset after 1 period
        :param T: int - number of periods
        '''

        self.initial_wealth = initial_wealth
        self.p = p
        self.a = a
        self.b = b
        self.risk_free = risk_free
        self.T = T
        self.env_id = env_id
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.num_episodes = num_episodes
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.minimal_size = minimal_size
        self.batch_size = batch_size
        self.sigma = sigma
        self.epsilon = epsilon

        self.env = gym.make(env_id, init_wealth=initial_wealth, p=p, A=a, B=b, riskfree=risk_free, T=T)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.return_list = ""
        self.action_list = ""
        self.whole_list = ""

    def asset_allocation_ddpg(self):

        '''
        This is the function to run asset allocation by using DDPG algorithm. Each step's action, reward, state,
        next state will be saved into df_full dataframe
        :return: None
        '''

        # Set seed for random numbers
        random.seed(0)
        np.random.seed(0)
        self.env.seed(0)
        torch.manual_seed(0)

        replay_buffer = rl_utils.ReplayBuffer(self.buffer_size)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_bound = self.env.action_space.high[0]  # maximum of the action
        # Set up the agent
        self.agent = DDPG(state_dim, self.hidden_dim, action_dim, action_bound, self.sigma, self.actor_lr, self.critic_lr,
                          self.tau, self.epsilon, self.gamma, self.device)
        # Train the model and save results into lists
        self.return_list, self.action_list, self.whole_list = rl_utils.train_off_policy_agent(self.env, self.agent,
                                                                                              self.num_episodes,
                                                                                              replay_buffer,
                                                                                              self.minimal_size,
                                                                                              self.batch_size)
        # Create a dataframe to store results of different episodes
        self.df_full = pd.DataFrame(self.whole_list, columns=['state', 'action', 'reward', 'next_state', 'done'],
                                    index=['step {}'.format(i) for i in range(self.T)] * self.num_episodes)

    def get_q_value(self):
        '''
        This function is used to calculate the Q value at each step. Q values of each step for every episode will be
        saved into the df_q_value dataframe
        :return: None
        '''

        if self.whole_list == "":
            # We check if we have run the asset allocation first, if not, we will run the asset allocation
            print("You haven't run asset allocation, running asset allocation now")
            self.asset_allocation_ddpg()
            self.get_q_value()
        else:
            # Calculate Q values
            q_idx = []
            for i in range(self.T):
                q_idx.append('Q value in T = {}'.format(i))
            self.df_q_value = pd.DataFrame(columns=q_idx)
            for t in range(self.T):
                q_value_list = []
                for idx, row in self.df_full.loc['step {}'.format(t)].iterrows():
                    q_value = row.reward + self.agent.gamma * (
                        self.agent.target_critic(torch.tensor(np.array([row.next_state])).to(torch.float32).view(-1, 1),
                                                 self.agent.target_actor(torch.tensor(
                                                     np.array([row.next_state])).to(torch.float32).view(-1,1)))) \
                              * (1 - row.done)
                    q_value_list.append(q_value)
                q_list = [x.item() for x in q_value_list]
                self.df_q_value['Q value in T = {}'.format(t)] = q_list

    def output(self, moving_window = 100):

        '''
        This function is used to output the final action list and Q value from different episodes after convergence.
        :param moving_window: int - number of episodes to use from the end to compute the average
        :return: action and Q values dataframes
        '''

        # Generate Q values
        self.get_q_value()

        # Reformat the action list from wide to long for easier computation
        self.df_action = self.df_full[['action']]
        self.df_action.reset_index(inplace=True)
        self.df_action.columns = ['step', 'action']
        self.df_action['episode'] = np.concatenate([([i]*self.T) for i in list(range(1, self.num_episodes + 1))], axis=0)
        self.df_action = pd.pivot(self.df_action, index = 'episode', columns = 'step', values = 'action')

        # Compute the mean of actions from final several episodes and output
        self.action = self.df_action.iloc[-moving_window:].mean()
        print(self.action)

        # Compute the mean of the Q values from final several episodes and output
        self.q_value = self.df_q_value.iloc[-moving_window:].mean()
        print(self.q_value)

        return self.action, self.q_value

    def plot_return_ma(self, moving_window = 100):

        '''
        This is simple function to plot moving average of final utility to visualize the convergence
        :param moving_window: int - number of episodes to use to compute the moving average
        :return: None
        '''

        ma_return = rl_utils.moving_average(self.return_list, moving_window)
        plt.plot(list(range(len(self.return_list)))[moving_window:], ma_return)
        plt.xlabel('Episodes')
        plt.ylabel('Final Utility')
        plt.title('Final Utility Moving Average (p = '
                  + str(self.p) + ', a = ' + str(self.a) + ', b = ' + str(self.b)
                  + ', T = ' + str(self.T) + ')')
        plt.show()


