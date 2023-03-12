from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym


class AssetAllocationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_wealth, p, A, B, riskfree, T):
        super(AssetAllocationEnv, self).__init__()
        self.init_wealth = init_wealth
        self.p = p
        self.A = A
        self.B = B
        self.riskfree_return = riskfree
        self.T = T
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=np.inf, dtype=np.float64, shape=(1,))

        # initialize state and time step
        self.state = None
        self.timestep = 0
        self.done = False

    def step(self, action):
        '''
        This function describes the state transition for a given action
        :param action: action to take
        :return: state, reward, whether it's completed
        '''
        assert self.action_space.contains(action), "Invalid action"
        if self.done:
            return self.state, reward, 0, True, {}
        xt = action
        if self.timestep == 0:
            wt = self.init_wealth
        else:
            wt = self.state
        if np.random.uniform() < self.p:
            risky_return = self.A
        else:
            risky_return = self.B
        wt1 = xt * wt * risky_return + (1 - xt) * wt * self.riskfree_return
        self.timestep += 1
        if self.timestep == self.T:
            self.done = True
            a = 1
            reward = (1 - np.exp(-a * wt1)) / a
            reward = reward.item()
        else:
            reward = 0
        self.state = wt1
        return self.state, reward, self.done, {}

    def reset(self):
        self.state = 1
        self.timestep = 0
        self.done = False
        return np.array([self.state], dtype=np.float64)

    def render(self, mode='human'):
        pass