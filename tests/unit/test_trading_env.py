import numpy as np
from models.trading_env import TradingEnv

def test_trading_env_initialization():
    """Tests if the environment initializes with the correct default values."""
    fake_data = np.random.rand(100, 5)
    env = TradingEnv(data=fake_data, initial_balance=5000)

    assert env.initial_balance == 5000
    assert env.balance == 5000
    assert env.shares_held == 0
    assert env.current_step == 0

def test_trading_env_buy_action():
    """Tests the logic of a 'buy' action."""
    # Data where the price is always 100
    fake_data = np.full((10, 5), 100)
    env = TradingEnv(data=fake_data, initial_balance=1000)
    env.reset()

    # Action 1 = Buy
    obs, reward, done = env.step(1)

    assert env.balance == 900  # 1000 - 100
    assert env.shares_held == 1
    # Net worth should be 900 (balance) + 1 * 100 (share value) = 1000
    assert env.net_worth == 1000

def test_trading_env_sell_action():
    """Tests the logic of a 'sell' action after a buy."""
    fake_data = np.full((10, 5), 100)
    env = TradingEnv(data=fake_data, initial_balance=1000)
    env.reset()

    # First, buy a share
    env.step(1)
    assert env.shares_held == 1
    assert env.balance == 900

    # Next, sell the share (Action 2 = Sell)
    env.step(2)
    assert env.shares_held == 0
    assert env.balance == 1000  # 900 + 100

def test_trading_env_hold_action():
    """Tests that a 'hold' action changes nothing."""
    fake_data = np.full((10, 5), 100)
    env = TradingEnv(data=fake_data, initial_balance=1000)
    env.reset()

    # Action 0 = Hold
    env.step(0)
    assert env.balance == 1000
    assert env.shares_held == 0
    assert env.net_worth == 1000