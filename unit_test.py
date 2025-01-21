import pytest
from Environments import TrainingEnv
from Environments import TestingEnv as ValidationEnv
import numpy as np
import pandas as pd


class TestTrainingEnv:

    @pytest.fixture
    def setup(self):
        data = pd.read_csv('KO_data.csv')
        data.drop(labels=['Date'], axis=1, inplace=True)
        data = data.iloc[:, :].copy(deep=True)
        return TrainingEnv(episode_length=2, data=data)
    
    def test_initialization(self, setup):
        env = setup
        assert env.budget == 10000
        assert env.portfolio_value == 10000
        assert env.cur_row_num == 0
        assert env.starting_row_num == 0
        assert env.asset_allocation == 0.0
        assert env.episode_length == 2
        assert env.cur_action == 2
        assert env.allocation_change == 0.0

    def test_get_obs(self, setup):
        env = setup
        array = np.array([0.08226522970933221,
                          0.8087919129312245,
                          0.614198272735933,
                          0.07637462123796929,
                          0.3548492373025801,
                          0.0])
        assert np.allclose(env._get_obs(), array) == True

    def test_get_info(self, setup):
        env = setup
        assert env._get_info() == {'Portfolio Value': 10000, 
                                   'Action Taken': 2, 
                                   'Asset Allocation': 0.0}

    def test_reset(self, setup):
        env = setup
        env.portfolio_value = 1500
        env.cur_action = 1
        env.starting_row_num = 5
        env.cur_row_num = 9
        env.asset_allocation = 2.0
        env.allocation_change = 0.5

        obs, info = env.reset(seed=5)

        assert env.portfolio_value == 10000
        assert env.cur_action == 2
        assert env.starting_row_num == 2318
        assert env.cur_row_num == 2318
        assert env.asset_allocation == 0.0
        assert env.allocation_change == 0.0

        array = np.array([0.0322325078173549,
                          0.42452658206995736,
                          0.357175589302116,
                          0.24570967083777226,
                          0.4111117340690847,
                          0.0])
        assert np.allclose(obs, array) == True

        assert info == {'Portfolio Value': 10000, 
                                   'Action Taken': 2, 
                                   'Asset Allocation': 0.0}


    def test_step(self, setup):
        env = setup
        obs, rew, terminated, truncated, info = env.step(3)
        assert env.cur_action == 3
        assert env.cur_row_num == 1
        assert env.asset_allocation == 0.1
        assert env.allocation_change == 0.1
        array = np.array([0.051949820380333196,
                          0.8039360338222462,
                          0.6808628032748046,
                          0.06603514842016697,
                          0.38048715906695335,
                          0.1])
        assert np.allclose(obs, array) == True
        assert rew == pytest.approx(-.0020646568)
        assert terminated == False
        assert truncated == False
        assert info == {'Portfolio Value': 9979.353431720145, 
                                   'Action Taken': 3, 
                                   'Asset Allocation': 0.1}
    
        

        obs, rew, terminated, truncated, info = env.step(0)
        assert env.cur_action == 0
        assert env.cur_row_num == 2
        assert env.asset_allocation == 0.0
        assert env.allocation_change == -0.1
        array = np.array([0.06691028589773108,
                          0.7945032774010329,
                          0.6723315861406489,
                          0.05872394755585882,
                          0.4033609876071836,
                          0.0])
        assert np.allclose(obs, array, atol=0.0001) == True
        assert rew == pytest.approx(-0.001)
        assert terminated == False
        assert truncated == False
        assert info == {'Portfolio Value': 9969.374078288425, 
                                   'Action Taken': 0, 
                                   'Asset Allocation': 0.0}

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
        assert env._get_new_portfolio_value() == pytest.approx(9979.353431720145)

    def test_get_reward(self, setup):
        env = setup
        env.cur_row_num = 1
        env.asset_allocation = 0.1
        env.cur_action = 3
        env.allocation_change = 0.1
        assert env._get_reward() == pytest.approx(-.0020646568)
        assert env.portfolio_value == pytest.approx(9979.353431720145)



class TestValidationEnv:

    @pytest.fixture
    def setup(self):
        data = pd.read_csv('KO_data.csv')
        data.drop(labels=['Date'], axis=1, inplace=True)
        data = data.iloc[0:3, :].copy(deep=True)
        return ValidationEnv(data=data)
    
    def test_reset(self, setup):
        env = setup
        env.portfolio_value = 1500
        env.cur_action = 1
        env.starting_row_num = 5
        env.cur_row_num = 9
        env.asset_allocation = 2.0
        env.allocation_change = 0.5

        obs, info = env.reset(seed=5)

        assert env.portfolio_value == 10000
        assert env.cur_action == 2
        assert env.starting_row_num == 0
        assert env.cur_row_num == 0
        assert env.asset_allocation == 0.0
        assert env.allocation_change == 0.0

        array = np.array([0.08226522970933221,
                          0.8087919129312245,
                          0.614198272735933,
                          0.07637462123796929,
                          0.3548492373025801,
                          0.0])
        assert np.allclose(obs, array) == True

        assert info == {'Portfolio Value': 10000, 
                                   'Action Taken': 2, 
                                   'Asset Allocation': 0.0}
        

    def test_step(self, setup):
        env = setup
        obs, rew, terminated, truncated, info = env.step(3)
        assert env.cur_action == 3
        assert env.cur_row_num == 1
        assert env.asset_allocation == 0.1
        assert env.allocation_change == 0.1
        array = np.array([0.051949820380333196,
                          0.8039360338222462,
                          0.6808628032748046,
                          0.06603514842016697,
                          0.38048715906695335,
                          0.1])
        assert np.allclose(obs, array) == True
        assert rew == pytest.approx(-.0020646568)
        assert terminated == False
        assert truncated == False
        assert info == {'Portfolio Value': 9979.353431720145, 
                                   'Action Taken': 3, 
                                   'Asset Allocation': 0.1}
    
        

        obs, rew, terminated, truncated, info = env.step(0)
        assert env.cur_action == 0
        assert env.cur_row_num == 2
        assert env.asset_allocation == 0.0
        assert env.allocation_change == -0.1
        array = np.array([0.06691028589773108,
                          0.7945032774010329,
                          0.6723315861406489,
                          0.05872394755585882,
                          0.4033609876071836,
                          0.0])
        assert np.allclose(obs, array, atol=0.0001) == True
        assert rew == pytest.approx(-0.001)
        assert terminated == True
        assert truncated == False
        assert info == {'Portfolio Value': 9969.374078288425, 
                                   'Action Taken': 0, 
                                   'Asset Allocation': 0.0}
