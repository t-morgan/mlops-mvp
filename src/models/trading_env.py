import numpy as np


# A simple trading environment for the DQL agent
class TradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.n_steps = len(data)
        self.n_features = data.shape[1]

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        # The state is the market data at the current time step
        return self.data[self.current_step]

    def step(self, action):
        # Actions: 0=Hold, 1=Buy, 2=Sell
        current_price = self.data[
            self.current_step, 0
        ]  # Assume first column is the main price

        # Execute action
        if action == 1:  # Buy
            # Buy one share
            if self.balance >= current_price:
                self.balance -= current_price
                self.shares_held += 1
        elif action == 2:  # Sell
            # Sell one share
            if self.shares_held > 0:
                self.balance += current_price
                self.shares_held -= 1

        # Update portfolio
        self.net_worth = self.balance + self.shares_held * current_price

        # Calculate reward (change in net worth)
        reward = self.net_worth - (
            self.balance + self.shares_held * self.data[self.current_step - 1, 0]
            if self.current_step > 0
            else self.initial_balance
        )

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps

        obs = self._get_observation() if not done else np.zeros(self.n_features)

        return obs, reward, done
