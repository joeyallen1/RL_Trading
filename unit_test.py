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
        return TrainingEnv(episode_length=2, data=data)
    
    def test_initialization(self, setup):
        env = setup
        assert env.budget == 10000
        assert env.portfolio_value == 10000
        assert env.cur_row_num == 0
        assert env.starting_row_num == 0
        assert env.asset_allocation == 0.0
        assert env.episode_length == 2
        assert env.cur_action == 0
        assert env.allocation_change == 0.0

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


    def test_step(self, setup):
        env = setup
        obs, rew, terminated, truncated, info = env.step(3)
        assert env.cur_action == 3
        assert env.cur_row_num == 1
        assert env.asset_allocation == 0.1
        assert env.allocation_change == 0.1
        array = np.array([0.00014867929282713304,
                                0.03335066736002774,
                                0.6450050877556867,
                                0.4592888893984638,
                                0.32254884209612456, 0.1])
        assert np.allclose(obs, array) == True
        assert rew == pytest.approx(-.0681225854)
        assert terminated == False
        assert truncated == False
        assert info == {'Portfolio Value': 9318.774145808777, 
                                   'Action Taken': 3, 
                                   'Asset Allocation': 0.1}
    
        

        obs, rew, terminated, truncated, info = env.step(0)
        assert env.cur_action == 0
        assert env.cur_row_num == 2
        assert env.asset_allocation == 0.0
        array = np.array([0.0003283315773097594,
                          0.03177904893973537,
                          0.6254686372437431,
                          0.32753872517122085,
                          0.3179901400175601, 0.0])
        assert np.allclose(obs, array) == True
        assert rew == pytest.approx(-.001)
        assert terminated == False
        assert truncated == False
        assert info == {'Portfolio Value': 9309.455371662967, 
                                   'Action Taken': 0, 
                                   'Asset Allocation': 0.0}
        assert env.allocation_change == -.1

        obs, rew, terminated, truncated, info = env.step(2)
        assert terminated == True

        

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
        env.allocation_change = 0.1
        assert env._get_new_portfolio_value() == pytest.approx(9318.77146)

    def test_get_reward(self, setup):
        env = setup
        env.cur_row_num = 1
        env.asset_allocation = 0.1
        env.cur_action = 3
        env.allocation_change = 0.1
        assert env._get_reward() == pytest.approx(-.0681225854)
        assert env.portfolio_value == pytest.approx(9318.77146)



# class TestValidationEnv:

#     @pytest.fixture
#     def setup(self):
#         data = pd.read_csv('Amazon Data.csv')
#         data.drop(labels=['Date'], axis=1, inplace=True)
#         data = data.iloc[0:int(0.7 * len(data)), :].copy(deep=True)
#         return ValidationEnv(episode_length=3, data=data)
    