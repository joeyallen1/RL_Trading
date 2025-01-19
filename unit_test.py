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
    def setup(self):
        data = pd.read_csv('Amazon Data.csv')
        data.drop(labels=['Date'], axis=1, inplace=True)
        data = data.iloc[0:int(0.7 * len(data)), :].copy(deep=True)
        return TrainingEnv(episode_length=3, data=data)
    
    def test_initialization(self, setup):
        env = setup
        assert env.budget == 10000
        assert env.portfolio_value == 10000
        assert env.cur_row_num == 0
        assert env.starting_row_num == 0
        assert env.asset_allocation == 0.0
        assert env.episode_length == 3
        assert env.cur_action == 0

    def test_get_obs(self, setup):
        env = setup
        array = np.array([0.0004522231894523374,
                                           0.022337782400184896,
                                           0.665141206940271,
                                           0.5709102556338256,
                                           0.35495395092346826,
                                           0.0])
        assert np.allclose(env._get_obs(), array) == True

    def test_get_info(self, setup):
        env = setup
        assert env._get_info() == {'Portfolio Value': 10000, 
                                   'Action Taken': 0, 
                                   'Asset Allocation': 0.0}

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
