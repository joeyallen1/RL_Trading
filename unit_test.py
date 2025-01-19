import pytest
from Environments import TrainingEnv
from Environments import TestingEnv as ValidationEnv
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.spaces import Box
import numpy as np
import pandas as pd


class TestTrainingEnv:

    @pytest.fixture
    def setup_data(self):
        data = pd.read_csv('Amazon Data.csv')
        data.drop(labels=['Date'], axis=1, inplace=True)
        data = data.iloc[0:int(0.7 * len(data)), :].copy(deep=True)
        return TrainingEnv(episode_length=3, data=data)
    
    def test_initialization(self):
        # assert env.budget == 10000
        pass

    def test_get_obs(self):
        pass

    def test_get_info(self):
        pass

    def test_reset(self):
        pass

    def test_step(self):
        pass

    def test_action_to_allocation(self):
        pass

    def test_get_new_portfolio_value(self):
        pass

    def test_get_reward(self):
        pass
