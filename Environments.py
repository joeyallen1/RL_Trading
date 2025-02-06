import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import numpy as np
import pandas as pd


class TrainingEnv(gym.Env):
    """An environment for training a reinforcement learning algorithm on stock data that implements
    OpenAI's gymnasium interface. The purpose of the algorithm that uses this environment 
    is to learn how to balance budget allocation between a single stock and cash reserves to maximize profits.
    The environment allows for changing the budget allocation at each step and provides a reward 
    in the form of the return at each step relative to the buy-and-hold strategy. """


    def __init__(self, training_data: pd.DataFrame, episode_length: int, budget: int =10000):
        """
        Args:
            training_data: dataframe of training data. Should contain columns for Close, MACD, 
            MACD Percentage, Volume Oscillator, CV, RSI, Pct Change.

            episode_length: the number of steps in each training episode. It is assumed that 
            the episode length is equal to the length of the validation and testing datasets
            that this environment will be used for (to allow for better generalization).

            budget: the starting budget allocated to the stock, set to 10000 by default
        """

        self.starting_budget = budget
        self.portfolio_value = budget
        self.buy_and_hold_value = budget
        self.cur_row_num = 0
        self.starting_row_num = 0
        self.asset_allocation = 1.0
        self.data = training_data   #Close, MACD, MACD Percentage, Volume Oscillator, CV, RSI, Pct Change
        self.episode_length = episode_length
        self.cur_action = 2
        self.allocation_change = 0.0

        
    
        # action space: Sell 25%, sell 10%, no change, buy 10%, buy 25% (percentages are of total portfolio value, asset + cash, at each timestep)
        self.action_space = Discrete(5)

        # observation space: MACD Percentage, Volume Oscillator, CV, RSI, Pct Change, Asset Allocation
        self.observation_space = Box(low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float64)


    def _get_obs(self) -> np.array:
        """Returns the state space at the current step, which consists of the indicators
        in the given data with the current asset allocation appended."""

        obs = np.array(self.data.iloc[self.cur_row_num, 2:])
        obs = np.append(obs, self.asset_allocation)
        return obs


    def _get_info(self) -> dict:
        """Returns a dictionary containing the current portfolio value, the action just taken, the current asset allocation,
        what the current portfolio value would be if 100% was allocated to the stock at every timestep,
        the regular MACD value at this timestep, and the RSI value at this timestep."""

        return {'Portfolio Value': self.portfolio_value, 'Action Taken': self.cur_action, 'Asset Allocation': self.asset_allocation,
                'Buy and Hold Value': self.buy_and_hold_value, 'MACD': self.data.iloc[self.cur_row_num, 1],
                'RSI': self.data.iloc[self.cur_row_num, 5]}


    def reset(self, seed: int =None) -> tuple:
        """Resets the environment to start the next training episode. This involves seeding the random generator field,
        resetting the current action, starting the next episode at a random location in the training data, resetting 
        the asset allocation, and resetting the portfolio and buy and hold portfolio values.
        Returns the next observation and extra state info."""

        super().reset(seed=seed)
        self.portfolio_value = self.starting_budget
        self.buy_and_hold_value = self.starting_budget
        self.cur_action = 2

        self.starting_row_num = self.np_random.integers(0, int(len(self.data)) - self.episode_length - 1)

        self.cur_row_num = self.starting_row_num
        self.asset_allocation = 1.0
        self.allocation_change = 0.0
        
        return self._get_obs(), self._get_info()


    def step(self, action: int) -> tuple:
        """Moves the environment forward one step using the provided action choice. Terminates if the 
        episode length has been reached. Records the new asset allocation and allocation change. Returns
        the next observation, the reward from the action taken, whether the episode is terminated
        or truncated, and extra info. """

        self.cur_action = action
        self.cur_row_num += 1
        if (self.cur_row_num - self.starting_row_num) > self.episode_length - 2:
            terminated = True
        else:
            terminated = False
        truncated = False
        self.allocation_change = self._action_to_allocation(action) - self.asset_allocation
        self.asset_allocation = self._action_to_allocation(action)
        self.update_buy_and_hold()
        obs = self._get_obs()
        rew = self._get_reward()
        info = self._get_info()
        return obs, rew, terminated, truncated, info
    
    def update_buy_and_hold(self):
        """Updates the value of the portfolio consisting of 100% allocation
        to the stock."""

        percent_change = self.data.iloc[self.cur_row_num, 0] - self.data.iloc[self.cur_row_num-1, 0]
        percent_change /= self.data.iloc[self.cur_row_num-1, 0]
        self.buy_and_hold_value *= (1 + percent_change)

    
    def _action_to_allocation(self, action: int) -> float:
        """Converts the given action to a new asset allocation. Depending on the action,
        the asset allocation is decreased, kept the same, or increased. The allocation is
        clipped from 0.0 to 1.0. Returns the new asset allocation."""

        alloc_change = 0.0
        if action == 0: alloc_change = -.25
        elif action == 1: alloc_change = -.1
        elif action == 2: alloc_change = 0.0
        elif action == 3: alloc_change = .1
        else: alloc_change = 0.25
        return max(0.0, min(1.0, self.asset_allocation + alloc_change))
    

    def _get_new_portfolio_value(self) -> float:
        """Calculates the new portfolio value (it assumes a step was just taken), taking into account the 
        change in close price of the stock and also keeping the cash kept in reserve at a constant price. Accounts for 
        transaction fees and/or slippage during trading by applying a fixed 1% cost to the price of the trade just made.
        Returns the new portfolio value."""

        percent_change = (self.data.iloc[self.cur_row_num, 0] - self.data.iloc[self.cur_row_num-1, 0]) / self.data.iloc[self.cur_row_num-1, 0]
        new_portfolio_value = self.portfolio_value * (self.asset_allocation * (1.0 + percent_change) + (1.0 - self.asset_allocation))
        new_portfolio_value = new_portfolio_value - (.01 * abs(self.allocation_change) * self.portfolio_value)
        return new_portfolio_value
    

    def _get_reward(self) -> float:
        """Calculates the reward for the last action taken. The reward function is the log return
        of the environment's portfolio minus the log return of just the stock price over the same 
        timestep. This is used to encourage the trading algorithm to outperform the stock and creates 
        a reward structure that gives more meaningful "feedback" on the actions being taken."""

        new_portfolio_value = max(self._get_new_portfolio_value(), 1.0)
        reward = np.log(new_portfolio_value / self.portfolio_value)
        self.portfolio_value = new_portfolio_value
        # return max(-2.0, min(2.0, reward))
        return reward - np.log(self.data.iloc[self.cur_row_num, 0] / self.data.iloc[self.cur_row_num-1, 0])
    






class TestingEnv(TrainingEnv):
    """An environment for stock trading that uses OpenAI's gymnasium interface. This
    class is used for testing a trained algorithm. The only method that is overriden
    is the reset method, since during testing the environment should be reset 
    to the start of the testing data rather than a random index (which is 
    used for training)."""


    def __init__(self, testing_data: pd.DataFrame):
        """Initializes the testing environment. Uses the length
        of the provided dataset as the episode length."""

        TrainingEnv.__init__(self, testing_data, len(testing_data))



    def reset(self, seed: int =None) -> tuple:
        """Resets the environment to start the testing episode. This involves seeding the random generator field,
        resetting the current action, starting the episode at the beginning of the testing set, resetting 
        the asset allocation, and resetting both the portfolio value and the buy and hold portfolio value.
        Returns the next observation and extra state info."""

        gym.Env.reset(self, seed=seed)
        self.portfolio_value = self.starting_budget
        self.buy_and_hold_value = self.starting_budget
        self.cur_action = 2

        self.starting_row_num = 0

        self.cur_row_num = self.starting_row_num

        self.asset_allocation = 1.0
        self.allocation_change = 0.0
        
        return self._get_obs(), self._get_info()