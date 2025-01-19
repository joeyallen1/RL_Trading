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

    def test_reset(self, setup):
        env = setup
        env.portfolio_value = 1500
        env.cur_action = 1
        env.starting_row_num = 5
        env.cur_row_num = 9
        env.asset_allocation = 2.0

        obs, info = env.reset(seed=5)

        assert env.portfolio_value == 10000
        assert env.cur_action == 0
        assert env.starting_row_num == 3217
        assert env.cur_row_num == 3217
        assert env.asset_allocation == 0.0

        array = np.array([0.13707649239765515,
                          0.4038240788890815,
                          0.33211783944424916,
                          0.462807681391612,
                          0.11893703950449101, 
                          0.0])
        assert np.allclose(obs, array) == True

        assert info == {'Portfolio Value': 10000, 
                                   'Action Taken': 0, 
                                   'Asset Allocation': 0.0}


    def test_step(self):
        pass

    def test_action_to_allocation(self, setup):
        env = setup
        env.asset_allocation = 0.5
        assert env._action_to_allocation(0) == .25
        assert env._action_to_allocation(1) == .4
        assert env._action_to_allocation(2) == .5
        assert env._action_to_allocation(3) == .6
        assert env._action_to_allocation(4) == .75

        env.asset_allocation = 0.1
        assert env._action_to_allocation(0) == 0.0
        env.asset_allocation = 0.9
        assert env._action_to_allocation(3) == 1.0
        assert env._action_to_allocation(4) == 1.0

    def test_get_new_portfolio_value(self, setup):
        env = setup
        env.cur_row_num = 1
        env.asset_allocation = 0.1
        env.cur_action = 3
        assert env._get_new_portfolio_value() == pytest.approx(9318.77146)

    def test_get_reward(self):
        pass
