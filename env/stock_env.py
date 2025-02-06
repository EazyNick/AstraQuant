# 주식 거래 환경을 정의하는 클래스

import numpy as np
import gym
from gym import spaces

import os
import sys

current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..")) # 현재 디렉토리에 따라 이 부분 수정
sys.path.append(project_root)

from manage import PathManager
path_manager = PathManager()

# 원하는 경로 추가
sys.path.append(path_manager.get_path("config"))
sys.path.append(path_manager.get_path("logs"))

# import
try:
    from logs import log_manager
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")

class StockTradingEnv(gym.Env):
    def __init__(self, stock_data):
        super(StockTradingEnv, self).__init__()
        self.initial_balance = config_manager.get_initial_balance()
        self.observation_window = config_manager.get_observation_window()
        self.feature_dim = stock_data.shape[1] # 입력 데이터의 feature 개수 자동 설정
        self.stock_data = stock_data
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        
        self.action_space = spaces.Discrete(3)  # 0: 매도, 1: 보유, 2: 매수
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_window, self.feature_dim), dtype=np.float32)

    def reset(self):
        """ 환경을 초기화하고 초기 상태를 반환 """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self.stock_data[self.current_step:self.current_step + self.observation_window] # (observation_window 값, feature_dim) 크기의 배열 반환

    def step(self, action):
        """ 액션을 실행하고 새로운 상태, 보상, 종료 여부 반환 """
        price = self.stock_data[self.current_step, 0]
        reward = 0

        if action == 2:  # 매수
            shares_to_buy = self.balance // price
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * price

        elif action == 0:  # 매도
            self.balance += self.shares_held * price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - self.observation_window
        next_state = self.stock_data[self.current_step:self.current_step + self.observation_window]

        if done:
            reward = self.balance

        return next_state, reward, done, {}

if __name__ == "__main__":
    stock_data = np.random.randn(60, 5)
    env = StockTradingEnv(stock_data)
    state = env.reset()

    print(f"초기 상태 shape: {state.shape}")

    done = False
    step_count = 0

    while not done:
        next_state, reward, done, _ = env.step(2)  # 매수 (Buy)
        step_count += 1
        print(f"🔹 Step: {step_count}, 다음 상태 shape: {next_state.shape}, 보상: {reward}, 종료 여부: {done}")

    print("✅ 환경 종료!")
