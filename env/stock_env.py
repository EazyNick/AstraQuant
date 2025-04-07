# 주식 거래 환경을 정의하는 클래스

import numpy as np
import gym
from gym import spaces
import torch
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, stock_data, writer=None):
        super(StockTradingEnv, self).__init__()
        self.writer = writer
        self.device = config_manager.get_device()
        self.initial_balance = config_manager.get_initial_balance()
        self.observation_window = config_manager.get_observation_window()
        self.transaction_fee = config_manager.get_transaction_fee() 
        self.feature_dim = stock_data.shape[1] # 입력 데이터의 feature 개수 자동 설정
        self.stock_data = stock_data
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0 # 보유 주식 수
        self.previous_portfolio_value = self.initial_balance 
        
        self.max_shares_per_trade = config_manager.get_max_shares_per_trade()
        self.action_space = spaces.Discrete(1 + 2 * self.max_shares_per_trade)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_window, self.feature_dim), dtype=np.float32)

        # ✅ TensorBoard 추가
        self.train_step = 0  # 학습 스텝 카운트
        self.total_reward = 0  # 최종 보상 추적용 변수

    def normalize_reward(self, value, scale=50000):
        value = torch.tensor(value, dtype=torch.float32).to(self.device)
        sign = torch.sign(value)  # 값의 부호 유지
        return sign * torch.log1p(abs(value) / scale) * scale  # log(1 + |value|) 방식

    def reset(self):
        """ 환경을 초기화하고 초기 상태를 반환 """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.previous_portfolio_value = self.initial_balance 

        # 🔹 기존 상태 (주가 데이터) + 보유 주식 수 추가
        state = self.stock_data[self.current_step:self.current_step + self.observation_window]
        shares_held_feature = np.full((self.observation_window, 1), self.shares_held)  # 보유 주식수를 feature로 추가
        state_with_shares = np.hstack((state, shares_held_feature))  # 상태 확장
        
        return state_with_shares

    def step(self, action):
        """ 액션을 실행하고 새로운 상태, 보상, 종료 여부 반환 """
        reward = 0
        price = self.stock_data[self.current_step, 0]
        if np.isnan(price) or price <= 0:
            log_manager.logger.warning(f"[Step {self.current_step}] 경고: 유효하지 않은 가격 {price}.")
            return None, 0, True  # 가격이 NaN이면 종료

        if action == 0:
            # 관망
            pass

        elif 1 <= action <= self.max_shares_per_trade: 
            # 매수 (Buy)
            shares_to_buy = action # action개 만큼 매수
            cost = shares_to_buy * price * (1 + self.transaction_fee)  # 거래 수수료 포함
            if cost <= self.balance:  # 잔고가 충분한 경우에만 매수
                self.shares_held += shares_to_buy
                self.balance -= cost
            else:
                reward -= 1  # 매수를 원했지만 실패한 경우 패널티 추가

        elif self.max_shares_per_trade < action <= 2 * self.max_shares_per_trade:
            # 매도 (Sell)
            if self.shares_held > 0: 
                shares_to_sell = action - self.max_shares_per_trade
                shares_to_sell = min(shares_to_sell, self.shares_held)
                revenue = shares_to_sell * price * (1 - self.transaction_fee)  # 거래 수수료 포함
                self.balance += revenue
                self.shares_held -= shares_to_sell # 매도한만큼 주식수량 조정
            else:
                reward -= 1  # 매도를 원했지만 실패한 경우 패널티 추가

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - self.observation_window
        next_state = self.stock_data[self.current_step:self.current_step + self.observation_window]

        # 보유 주식 수 히스토리를 저장하는 배열 추가
        if not hasattr(self, "shares_held_history"):
            self.shares_held_history = np.zeros(self.observation_window)

        # 가장 오래된 값을 제거하고, 새로운 보유 주식 수 추가
        self.shares_held_history = np.roll(self.shares_held_history, shift=-1)
        self.shares_held_history[-1] = self.shares_held  # 최신 보유 주식 수 업데이트

        # 과거 보유 주식 수 기록을 상태와 함께 결합
        shares_held_feature = self.shares_held_history.reshape(-1, 1)  # (observation_window, 1)
        next_state_with_shares = np.hstack((next_state, shares_held_feature / 100))

        # 새로운 포트폴리오 가치 계산
        new_portfolio_value = self.balance + (self.shares_held * price)

        short_term_reward = 0
        long_term_reward = 0
        holding_reward = 0
        future_reward = 0
        future_return = 0

        # 포트폴리오 가치 변화율을 보상으로 설정 (수익률 기반 보상), 단기 수익률 보상
        if self.previous_portfolio_value > 0:
            short_term_reward = ((new_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value) * 100 * 2
        else:
            short_term_reward = 0

        # 장기적 보상을 반영하도록 강화 (현재 가치 대비 초기 가치)
        long_term_reward = ((new_portfolio_value - self.initial_balance) / self.initial_balance) * 100 * 1

        # 보유 주식 가격 상승 시 추가 보상
        if self.shares_held > 0 and self.current_step > 0:
            holding_reward = (price - self.stock_data[self.current_step - 1, 0]) * self.shares_held * 1
        else:
            holding_reward = 0

        # 18일 후의 `Buy & Hold` 수익률 계산
        future_step = min(self.current_step + 10, len(self.stock_data) - 1)
        # 현재 스텝을 제외한 5일 이내의 최고가 & 최저가 찾기
        future_max_price = np.max(self.stock_data[self.current_step + 1:future_step + 1, 0])
        future_min_price = np.min(self.stock_data[self.current_step + 1:future_step + 1, 0])
        
        # 리워드 계산
        if 1 <= action <= self.max_shares_per_trade:  # 매수(Buy)
            if future_max_price <= price:  # 미래 최고가가 현재 가격보다 낮거나 같으면 손실 가능성이 큼
                future_return = ((future_min_price - price) / price) * self.shares_held * 1.5
            else:  # 미래 최고가가 현재 가격보다 높으면 기존 방식 유지
                future_return = ((future_max_price - price) / price) * self.shares_held * 1.2
        elif self.max_shares_per_trade < action <= 2 * self.max_shares_per_trade:  # 매도(Sell)
            if future_min_price >= price:  # 미래 최저가가 현재 가격보다 높거나 같으면 손실 가능성이 큼
                future_return = ((price - future_max_price) / price) * self.shares_held * 1.2
            else:  # 미래 최저가가 현재 가격보다 낮으면 기존 방식 유지
                future_return = ((price - future_min_price) / price) * self.shares_held * 1.5

        elif action == 0:  # 관망(Hold)
            if self.shares_held > 0:  # 주식을 보유 중이라면
                # 미래 최고가와 현재 가격 비교
                if future_max_price > price:  # 가격이 오를 경우 큰 보상
                    future_return = ((future_max_price - price) / price) * self.shares_held * 1.2  # 가격 상승 보상
                else:  # 가격이 떨어지거나 그대로인 경우 패널티
                    future_return = ((future_min_price - price) / price) * self.shares_held * 2  # 패널티는 더 크게 (음수값)
            else:  # 주식을 보유하지 않은 상태라면
                # 미래 가격이 오르면 주식을 사지 않은 것에 대한 패널티
                if future_max_price > price:
                    future_return = -((future_max_price - price) / price) * 2  # 매수 기회를 놓친 것에 대한 패널티
                # 미래 가격이 떨어지면 주식을 사지 않은 것에 대한 보상
                else:
                    future_return = ((price - future_min_price) / price) * 1.2  # 하락을 피한 것에 대한 보상
                
        future_reward = future_return * 5  # 수익률 기반 보상

        # ✅ 최종 보상 (각 보상 요소를 합산)
        reward = short_term_reward + long_term_reward + holding_reward + future_reward + reward
        self.total_reward += reward  # ✅ 누적 보상 업데이트

        # # 🔹 **액션 로그 및 상태 변화 확인**
        # log_manager.logger.debug(
        #     f"[Step {self.current_step}] Action: {['Sell', 'Hold', 'Buy'][action]}, Price: {price:.2f}, "
        #     f"Prev Balance: {previous_portfolio_value:.2f} → {self.balance:.2f}, "
        #     f"Prev Shares: {previous_shares_held} → {self.shares_held}, "
        #     f"Portfolio: {self.previous_portfolio_value:.2f} → {new_portfolio_value:.2f}, "
        #     f"Reward: {reward:.4f}"
        # )

        self.train_step += 1  # 학습 스텝 증가 
        # ✅ TensorBoard 기록
        self.writer.add_scalar("Portfolio Value", new_portfolio_value, self.train_step)
        self.writer.add_scalar("Shares Held", self.shares_held, self.train_step)
        self.writer.add_scalar("Reward/Short-Term", short_term_reward, self.train_step)
        self.writer.add_scalar("Reward/Long-Term", long_term_reward, self.train_step)
        self.writer.add_scalar("Reward/Future", future_reward, self.train_step)
        self.writer.add_scalar("Reward/Total", reward, self.train_step)

        # ✅ 보상 정규화 적용
        reward = self.normalize_reward(reward)

        self.previous_portfolio_value = new_portfolio_value
         
        # log_manager.logger.debug(f"Step: {self.current_step}, Action: {['Sell', 'Hold', 'Buy'][action]}, Reward: {reward}, Portfolio: {new_portfolio_value}, Shares Held: {self.shares_held}")

        return next_state_with_shares, reward, done

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
