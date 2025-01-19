import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import numpy as np
import pandas as pd

class TrainingEnv(gym.Env):

    def __init__(self, data, episode_length = 250, budget=10000):
        self.budget = budget
        self.portfolio_value = budget
        self.cur_row_num = 0
        self.starting_row_num = 0
        self.asset_allocation = 0.0
        self.data = data
        self.episode_length = episode_length
        self.cur_action = 0
    
        # action space: Sell 25%, sell 10%, no change, buy 10%, buy 25% (percentages are of total portfolio value, asset + cash, at each timestep)
        self.action_space = Discrete(5)

        # observation space: Close, Volume, SMA Ratio, RSI, Bandwidth, Asset Allocation
        self.observation_space = Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, 100.0, np.inf, 1.0]), dtype=np.float64)

    # returns the current row in dataframe with current asset allocation appended
    def _get_obs(self):
        obs = np.array(self.data.iloc[self.cur_row_num, :])
        obs = np.append(obs, self.asset_allocation)
        return obs

    # returns current portfolio value
    def _get_info(self):
        return {'Portfolio Value': self.portfolio_value, 'Action Taken': self.cur_action, 'Asset Allocation': self.asset_allocation}

    # sets the starting row and starting asset allocation
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.portfolio_value = self.budget
        self.cur_action = 0

        self.starting_row_num = self.np_random.integers(0, int(len(self.data)) - self.episode_length - 1)

        self.cur_row_num = self.starting_row_num
        self.asset_allocation = 0.0
        
        return self._get_obs(), self._get_info()

    # moves to the next row in data, updates reward and current portfolio value
    def step(self, action):
        self.cur_action = action
        self.cur_row_num += 1
        if (self.cur_row_num - self.starting_row_num) > self.episode_length:
            terminated = True
        else:
            terminated = False
        truncated = False
        self.asset_allocation = self._action_to_allocation(action)
        obs = self._get_obs()
        rew = self._get_reward()
        info = self._get_info()
        return obs, rew, terminated, truncated, info
    
    # converts action to asset allocation value
    def _action_to_allocation(self, action):
        allocation_change = 0.0
        if action == 0: allocation_change = -.25
        elif action == 1: allocation_change = -.1
        elif action == 2: allocation_change = 0.0
        elif action == 3: allocation_change = .1
        else: allocation_change = 0.25
        return max(0.0, min(1.0, self.asset_allocation + allocation_change))
    
    # calculates new portfolio value 
    # accounts for possible commision costs + slippage by applying a fixed 1% cost to the price of each trade
    def _get_new_portfolio_value(self):
        asset_change = (self.data.iloc[self.cur_row_num, 0] - self.data.iloc[self.cur_row_num - 1, 0]) / self.data.iloc[self.cur_row_num - 1, 0]
        new_portfolio_value = self.portfolio_value * (self.asset_allocation * (1.0 + asset_change) + (1.0 - self.asset_allocation))
        allocation_change = 0.0
        if self.cur_action == 0: allocation_change = .25
        elif self.cur_action == 1: allocation_change = .1
        elif self.cur_action == 2: allocation_change = 0.0
        elif self.cur_action == 3: allocation_change = .1
        else: allocation_change = 0.25
        new_portfolio_value = new_portfolio_value - (.01 * allocation_change * self.portfolio_value)
        return new_portfolio_value
    
    # returns reward in the form of regular percent return of the total portfolio (stock + cash) over this timestep
    def _get_reward(self):
        new_portfolio_value = self._get_new_portfolio_value()
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.portfolio_value = new_portfolio_value
        return reward 
    


class TestingEnv(TrainingEnv):

    def __init__(self, data):
        TrainingEnv.__init__(self, data)

    # sets the starting row and starting asset allocation
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.portfolio_value = self.budget
        self.cur_action = 0

        self.starting_row_num = 0

        self.cur_row_num = self.starting_row_num

        self.asset_allocation = 0.0
        
        return self._get_obs(), self._get_info()
    
    # moves to the next row in data, updates reward and current portfolio value
    def step(self, action):
        self.cur_action = action
        self.cur_row_num += 1
        if self.cur_row_num >= int(len(self.data)) - 1:
            terminated = True
        else:
            terminated = False
        truncated = False
        self.asset_allocation = self._action_to_allocation(action)
        obs = self._get_obs()
        rew = self._get_reward()
        info = self._get_info()
        return obs, rew, terminated, truncated, info