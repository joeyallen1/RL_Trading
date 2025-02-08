import pytest
from Environments import TrainingEnv
from Environments import TestingEnv as ValidationEnv
import numpy as np
import pandas as pd


class TestTrainingEnv:

    @pytest.fixture
    def setup(self) -> TrainingEnv:
        data = pd.read_csv('./KO_scaled/Testing.csv')
        data.drop(labels=['Date'], axis=1, inplace=True)
        data = data.iloc[:, :].copy(deep=True)
        return TrainingEnv(data, episode_length=3)
    
    def test_initialization(self, setup):
        env = setup
        assert env.starting_budget == 10000
        assert env.portfolio_value == 10000
        assert env.buy_and_hold_value == 10000
        assert env.cur_row_num == 0
        assert env.starting_row_num == 0
        assert env.asset_allocation == 1.0
        assert len(env.data) == 735
        assert env.episode_length == 3
        assert env.cur_action == 2
        assert env.allocation_change == 0.0

    def test_get_obs(self, setup):
        env = setup
        array = np.array([0.12106316078377355,
                          0.24456287821482703,
                          0.2936693945056019,
                          -0.08794532975319626,
                          -0.8982335401912381, 
                          1.0])
        assert np.allclose(env._get_obs(), array, rtol=0.01) == True
        
        env.cur_row_num = 1
        array = np.array([-0.25376864034875524,
                          0.6030791113455097,
                          1.507248668554219,
                          -0.5937054429455657,
                          -2.365991921602965, 
                          1.0])
        assert np.allclose(env._get_obs(), array, rtol=0.01) == True



    def test_get_info(self, setup):
        env = setup
        assert env._get_info() == {'Portfolio Value': 10000, 
                                   'Action Taken': 2, 
                                   'Asset Allocation': 1.0,
                                   'Buy and Hold Value': 10000,
                                   'MACD': 0.2254565100946166,
                                   'RSI': 50.05678725595876}

    def test_reset(self, setup):
        env = setup
        env.portfolio_value = 1500
        env.cur_action = 1
        env.starting_row_num = 5
        env.cur_row_num = 9
        env.buy_and_hold_value = 10100.0
        env.asset_allocation = 2.0
        env.allocation_change = 0.5

        obs, info = env.reset(seed=5)

        assert env.portfolio_value == 10000
        assert env.buy_and_hold_value == 10000
        assert env.cur_action == 2
        assert env.starting_row_num == 490
        assert env.cur_row_num == 490
        assert env.asset_allocation == 1.0
        assert env.allocation_change == 0.0

        array = np.array([-0.2070785 ,  0.3377747 , -0.6652482 , -0.98585003, -0.23207383, 
                          1.0])
        assert np.allclose(obs, array) == True

        assert info == {'Portfolio Value': 10000, 
                                   'Action Taken': 2, 
                                   'Asset Allocation': 1.0,
                                   'Buy and Hold Value': 10000,
                                   'MACD': 0.0046280243457346,
                                   'RSI': 32.55843718916388}


    def test_step(self, setup):
        env = setup
        obs, rew, terminated, truncated, info = env.step(0)
        assert env.buy_and_hold_value == pytest.approx(9603.797715)
        assert env.cur_action == 0
        assert env.cur_row_num == 1
        assert env.asset_allocation == .75
        assert env.allocation_change == -.25
        array = np.array([-0.25376864034875524,
                          0.6030791113455097,
                          1.507248668554219,
                          -0.5937054429455657,
                          -2.365991921602965, 
                          .75])
        assert np.allclose(obs, array, rtol=.01) == True
        assert rew == pytest.approx(0.0076809765 * 10)
        assert terminated == False
        assert truncated == False
        assert info == {'Portfolio Value': 9677.84828602265, 
                                   'Action Taken': 0, 
                                   'Asset Allocation': .75,
                                   'Buy and Hold Value': 9603.797714696868,
                                   'MACD': -0.0268350408070006,
                                   'RSI': 40.200543615505545}
    
        

        obs, rew, terminated, truncated, info = env.step(3)
        assert env.buy_and_hold_value == pytest.approx(9652.911751)
        assert env.cur_action == 3
        assert env.cur_row_num == 2
        assert env.asset_allocation == .85
        assert env.allocation_change == pytest.approx(.1)
        array = np.array([-0.5162393203401424,0.5714804258515582,2.0931656119780517,-0.8315609558397491,-2.4575531583820083,
                          .85])
        assert np.allclose(obs, array, rtol=.01) == True
        assert rew == pytest.approx(-.0017596596 * 10)
        assert terminated == True
        assert truncated == False
        assert info == {'Portfolio Value': 9710.239260837297, 
                                   'Action Taken': 3, 
                                   'Asset Allocation': .85,
                                   'Buy and Hold Value': 9652.911751474907,
                                   'MACD': -0.2023291387435577,
                                   'RSI': 35.565219822349206}

        

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
        assert env._get_new_portfolio_value() == pytest.approx(9950.379771)

        env.portfolio_value = 9950.379771
        env.cur_row_num = 2
        env.asset_allocation = 0.0
        env.cur_action = 1
        env.allocation_change = -.1
        assert env._get_new_portfolio_value() == pytest.approx(9940.429392)

    def test_get_reward(self, setup):
        env = setup
        env.cur_row_num = 1
        env.asset_allocation = 0.1
        env.cur_action = 3
        env.allocation_change = 0.1
        assert env._get_reward() == pytest.approx(0.0354521029 * 10)
        assert env.portfolio_value == pytest.approx(9950.379771)

    
    def test_update_buy_and_hold(self, setup):
        env = setup
        env.cur_row_num=1
        env._update_buy_and_hold()
        assert env.buy_and_hold_value == 9603.797714696868

    




class TestValidationEnv:

    @pytest.fixture
    def setup(self):
        data = pd.read_csv('./KO_scaled/Testing.csv')
        data.drop(labels=['Date'], axis=1, inplace=True)
        data = data.iloc[:, :].copy(deep=True)
        return ValidationEnv(data)
    
    def test_reset(self, setup):
        env = setup
        env.portfolio_value = 1500
        env.cur_action = 1
        env.starting_row_num = 5
        env.cur_row_num = 9
        env.buy_and_hold_value = 10100.0
        env.asset_allocation = 2.0
        env.allocation_change = 0.5

        obs, info = env.reset(seed=5)

        assert env.portfolio_value == 10000
        assert env.buy_and_hold_value == 10000
        assert env.cur_action == 2
        assert env.starting_row_num == 0
        assert env.cur_row_num == 0
        assert env.asset_allocation == 1.0
        assert env.allocation_change == 0.0

        array = np.array([0.12106316078377355,0.24456287821482703,0.2936693945056019,-0.08794532975319626,-0.8982335401912381, 
                          1.0])
        assert np.allclose(obs, array) == True

        assert info == {'Portfolio Value': 10000, 
                                   'Action Taken': 2, 
                                   'Asset Allocation': 1.0,
                                   'Buy and Hold Value': 10000,
                                   'MACD':  0.2254565100946166,
                                   'RSI': 50.05678725595876}