# 주식 거래 환경을 정의하는 클래스

import numpy as np
import gym
from gym import spaces
import random
import torch

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
        self.device = config_manager.get_device()
        self.initial_balance = config_manager.get_initial_balance()
        self.observation_window = config_manager.get_observation_window()
        self.transaction_fee = config_manager.get_transaction_fee() 
        self.epsilon = config_manager.get_epsilon()
        self.feature_dim = stock_data.shape[1] # 입력 데이터의 feature 개수 자동 설정
        self.stock_data = stock_data
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0 # 보유 주식 수
        self.previous_portfolio_value = self.initial_balance 
        self.epsilon_min = 0.01
        
        
        self.action_space = spaces.Discrete(3)  # 0: 매도, 1: 보유, 2: 매수
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_window, self.feature_dim), dtype=np.float32)

    def normalize_reward(self, value, scale=50000): # -1,000 ~ 1,000
        value = torch.tensor(value, dtype=torch.float32).to(self.device)  # Tensor 변환 후 GPU/CPU 이동
        return torch.tanh(value / scale) * scale  # 정규화 적용

    def reset(self):
        """ 환경을 초기화하고 초기 상태를 반환 """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.previous_portfolio_value = self.initial_balance 
        return self.stock_data[self.current_step:self.current_step + self.observation_window] # (observation_window 값, feature_dim) 크기의 배열 반환

    def step(self, action):
        """ 액션을 실행하고 새로운 상태, 보상, 종료 여부 반환 """
        price = self.stock_data[self.current_step, 0]

        # 입실론(ε) 값을 이용한 랜덤 액션 (탐색)
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1, 2])  # 0: 매도, 1: 보유, 2: 매수
            self.epsilon = max(self.epsilon * 0.999, self.epsilon_min)  # 0.999 → 지수적 감소

        if action == 2:  # 매수 (Buy)
            shares_to_buy = self.balance / (price * (1 + self.transaction_fee)) # 살 수 있는 최대 주식 수
            shares_to_buy = int(shares_to_buy) # 정수 값으로 변환 (소수점 이하 버림)
            cost = shares_to_buy * price * (1 + self.transaction_fee)  # 거래 수수료 포함
            if cost <= self.balance:  # 잔고가 충분한 경우에만 매수
                self.shares_held += shares_to_buy
                self.balance -= cost

        elif action == 0:  # 매도 (Sell)
            revenue = self.shares_held * price * (1 - self.transaction_fee)  # 거래 수수료 포함
            self.balance += revenue
            self.shares_held = 0  # 전량 매도

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - self.observation_window
        next_state = self.stock_data[self.current_step:self.current_step + self.observation_window]

        # 새로운 포트폴리오 가치 계산
        new_portfolio_value = self.balance + (self.shares_held * price)

        short_term_reward = 0
        long_term_reward = 0
        holding_reward = 0
        future_reward = 0
        reward = 0

        # 포트폴리오 가치 변화율을 보상으로 설정 (수익률 기반 보상), 단기 수익률 보상
        if self.previous_portfolio_value > 0:
            short_term_reward = ((new_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value) * 100 * 50
        else:
            short_term_reward = 0

        # 장기적 보상을 반영하도록 강화 (현재 가치 대비 초기 가치)
        long_term_reward = ((new_portfolio_value - self.initial_balance) / self.initial_balance) * 100 * 50

        # 보유 주식 가격 상승 시 추가 보상
        if self.shares_held > 0 and self.current_step > 0:
            holding_reward = (price - self.stock_data[self.current_step - 1, 0]) * self.shares_held * 2
        else:
            holding_reward = 0

        # 1달(30일) 후의 `Buy & Hold` 수익률 계산
        future_step = min(self.current_step + 30, len(self.stock_data) - 1)
        future_price = self.stock_data[future_step, 0]  # 가장 먼 미래의 데이터를 가져옴
        future_return = ((future_price - price) / price) * 100  # `Buy & Hold` 수익률 계산
        future_reward = future_return * 500  # 수익률 기반 보상

        # ✅ 최종 보상 (각 보상 요소를 합산)
        reward = short_term_reward + long_term_reward + holding_reward + future_reward

        # ✅ 보상 정규화 적용
        reward = self.normalize_reward(reward)

        self.previous_portfolio_value = new_portfolio_value  

        # log_manager.logger.debug(f"Step: {self.current_step}, Action: {['Sell', 'Hold', 'Buy'][action]}, Reward: {reward}, Portfolio: {new_portfolio_value}, Shares Held: {self.shares_held}")

        return next_state, reward, done, {}

if __name__ == "__main__":
    stock_data = np.random.randn(60, 5)
    env = StockTradingEnv(stock_data)
    state = env.reset()

    log_manager.logger.debug(f"초기 상태 shape: {state.shape}")

    done = False
    step_count = 0

    while not done:
        next_state, reward, done, _ = env.step(2)  # 매수 (Buy)
        step_count += 1
        log_manager.logger.debug(f"🔹 Step: {step_count}, 다음 상태 shape: {next_state.shape}, 보상: {reward}, 종료 여부: {done}")

    log_manager.logger.debug("✅ 환경 종료!")
