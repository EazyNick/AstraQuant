import numpy as np
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        
        self.action_space = spaces.Discrete(3)  # 0: 매도, 1: 보유, 2: 매수
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30, 1), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self.stock_data[self.current_step:self.current_step+30]

    def step(self, action):
        price = self.stock_data[self.current_step]
        reward = 0

        if action == 2:  # 매수
            shares_to_buy = self.balance // price
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * price

        elif action == 0:  # 매도
            self.balance += self.shares_held * price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 30
        next_state = self.stock_data[self.current_step:self.current_step+30]

        if done:
            reward = self.balance

        return next_state, reward, done, {}
