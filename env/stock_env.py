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
        
        # 🔥 Close 가격을 별도로 보관 (포트폴리오 계산용)
        self.close_prices = stock_data[:, 0]  # Close 가격만 별도 보관
        
        # 🔥 Close를 제외한 학습용 데이터 생성
        self.stock_data = np.delete(stock_data, 0, axis=1)  # 첫 번째 컬럼(Close) 제거
        
        self.feature_dim = self.stock_data.shape[1] # Close 제외한 feature 개수
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0 # 보유 주식 수
        self.close_price_scale = 10  # 'Close' 값을 원래 가격으로 복원할 때 사용할 스케일
        self.previous_portfolio_value = self.initial_balance
        
        # 🔥 주식 보유량 스케일링 개선
        self.max_shares_per_trade = config_manager.get_max_shares_per_trade()
        self.shares_scaling_factor = config_manager.get_shares_scaling_factor()  # 설정에서 가져오기
        
        self.action_space = spaces.Discrete(1 + 2 * self.max_shares_per_trade)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_window, self.feature_dim), dtype=np.float32)

        # ✅ TensorBoard 추가
        self.train_step = 0  # 학습 스텝 카운트
        self.total_reward = 0  # 최종 보상 추적용 변수

    def normalize_reward(self, value, scale=50000):
        value = torch.tensor(value, dtype=torch.float32).to(self.device)
        sign = torch.sign(value)  # 값의 부호 유지
        return sign * torch.log1p(abs(value) / scale) * scale  # log(1 + |value|) 방식

    def normalize_shares_for_learning(self, shares_held):
        """
        주식 보유량을 학습용 소수점 스케일로 변환
        
        Args:
            shares_held (int): 실제 보유 주식 수 (정수)
            
        Returns:
            float: 학습용 스케일된 주식 보유량 (0.XXX 형태)
        """
        return shares_held / self.shares_scaling_factor

    def reset(self):
        """ 환경을 초기화하고 초기 상태를 반환 """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0  # 정수값으로 초기화
        self.previous_portfolio_value = self.initial_balance 
        self.last_buy_step = None  # 🔥 매수 시점 추적 초기화

        # 🔹 기존 상태 (주가 데이터) + 보유 주식 수 추가
        state = self.stock_data[self.current_step:self.current_step + self.observation_window]
        
        # 🔥 주식 보유량을 학습용 스케일로 변환
        scaled_shares = self.normalize_shares_for_learning(self.shares_held)
        shares_held_feature = np.full((self.observation_window, 1), scaled_shares)  # 스케일된 보유 주식수를 feature로 추가
        state_with_shares = np.hstack((state, shares_held_feature))  # 상태 확장
        
        return state_with_shares

    def step(self, action):
        """ 액션을 실행하고 새로운 상태, 보상, 종료 여부 반환 """
        reward = 0
        price = self.close_prices[self.current_step] * self.close_price_scale  # 별도 보관된 Close 가격 사용
        if np.isnan(price) or price <= 0:
            log_manager.logger.warning(f"[Step {self.current_step}] 경고: 유효하지 않은 가격 {price}.")
            return None, 0, True  # 가격이 NaN이면 종료

        if action == 0:
            # 관망
            pass

        elif 1 <= action <= self.max_shares_per_trade: 
            # 매수 (Buy) - action * 30주 만큼 매수
            shares_to_buy = action * 30  # 액션 값에 30을 곱함
            cost = shares_to_buy * price * (1 + self.transaction_fee)  # 거래 수수료 포함
            
            # 🔥 디버깅: 매수 시도 상황 로그
            if self.train_step % 1000 == 0:  # 1000 스텝마다만 출력
                log_manager.logger.debug(
                    f"[Step {self.current_step}] 매수 시도:\n"
                    f"  - 액션: {action} (매수 {shares_to_buy}주)\n"
                    f"  - 현재 가격: {price:,.0f}원\n"
                    f"  - 필요 비용: {cost:,.0f}원\n"
                    f"  - 현재 잔고: {self.balance:,.0f}원\n"
                    f"  - 매수 가능: {'예' if cost <= self.balance else '아니오'}"
                )
            
            if cost <= self.balance:  # 잔고가 충분한 경우에만 매수
                self.shares_held += shares_to_buy
                self.balance -= cost
                if self.train_step % 1000 == 0:
                    portfolio_value = self.balance + (self.shares_held * price)
                    log_manager.logger.debug(f"  - 매수 성공! 보유 주식: {self.shares_held}주, 잔고: {self.balance:,.0f}원, 포트폴리오 밸류: {portfolio_value:,.0f}원")
                self.last_buy_step = self.current_step  # 🔥 매수 시점 업데이트
            else:
                reward -= 0.0001 * self.shares_held  # 매수 실패 패널티 감소
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(f"  - 매수 실패! 잔고 부족")

        elif self.max_shares_per_trade < action <= 2 * self.max_shares_per_trade:
            # 매도 (Sell) - (action - max_shares_per_trade) * 30주 만큼 매도
            if self.shares_held > 0: 
                shares_to_sell = (action - self.max_shares_per_trade) * 30  # 액션 값에 30을 곱함
                shares_to_sell = min(shares_to_sell, self.shares_held)
                revenue = shares_to_sell * price * (1 - self.transaction_fee)  # 거래 수수료 포함
                self.balance += revenue
                self.shares_held -= shares_to_sell # 매도한만큼 주식수량 조정
                
                # 🔥 디버깅: 매도 성공 로그
                if self.train_step % 1000 == 0:
                    portfolio_value = self.balance + (self.shares_held * price)
                    log_manager.logger.debug(
                        f"[Step {self.current_step}] 매도 성공:\n"
                        f"  - 액션: {action} (매도 {shares_to_sell}주)\n"
                        f"  - 매도 수익: {revenue:,.0f}원\n"
                        f"  - 보유 주식: {self.shares_held}주\n"
                        f"  - 잔고: {self.balance:,.0f}원\n"
                        f"  - 포트폴리오 밸류: {portfolio_value:,.0f}원"
                    )
            else:
                reward -= 0.01  # 매도 실패 패널티 증가 (보유 주식이 없는데 매도하려 할 때)
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(f"[Step {self.current_step}] 매도 실패! 보유 주식 없음")

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - self.observation_window
        next_state = self.stock_data[self.current_step:self.current_step + self.observation_window]

        # 새로운 포트폴리오 가치 계산
        new_portfolio_value = self.balance + (self.shares_held * price)

        # 1. 단기 수익률 보상
        short_term_reward = 0
        long_term_reward = 0
        holding_reward = 0
        risk_penalty = 0
        transaction_penalty = 0
        holding_encouragement = 0

        # 포트폴리오 가치 변화율을 보상으로 설정 (수익률 기반 보상), 단기 수익률 보상
        if self.previous_portfolio_value > 0:
            short_term_reward = ((new_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value) * 12

        # 2. 장기 수익률 보상 (현재 가치 대비 초기 가치)
        long_term_reward = ((new_portfolio_value - self.initial_balance) / self.initial_balance) * 10

        # 3. 보유 주식 가격 변화 보상
        if self.shares_held > 0 and self.current_step > 1:  # 🔥 첫 번째 스텝이 아닌 경우에만 계산
            # 🔥 current_step이 이미 증가된 상태이므로 -2를 해야 현재 스텝과 이전 스텝을 비교
            prev_price = self.close_prices[self.current_step - 2] * self.close_price_scale  # 별도 보관된 Close 가격 사용
            price_change = (price - prev_price) / prev_price
            holding_reward = price_change * self.shares_held * 0.1

        # 4. 매수 후 미래 가격 변화 보상 (새로 추가)
        future_price_reward = 0
        if hasattr(self, 'last_buy_step') and self.last_buy_step is not None:
            # 🔥 매수한 스텝 이후 몇 스텝이 지났는지 확인
            steps_since_buy = self.current_step - self.last_buy_step
            
            # 🔥 매수 후 1-5 스텝 동안의 가격 변화를 고려
            if 1 <= steps_since_buy <= 5:
                buy_price = self.close_prices[self.last_buy_step] * self.close_price_scale
                current_price = price  # 현재 스텝의 가격
                future_price_change = (current_price - buy_price) / buy_price
                
                # 🔥 매수 후 가격 상승 시 양의 보상, 하락 시 음의 보상
                # 시간이 지날수록 보상 가중치 감소 (즉시 반응을 더 중요하게)
                time_weight = 1.0 / steps_since_buy  # 1스텝 후: 1.0, 2스텝 후: 0.5, 3스텝 후: 0.33...
                future_price_reward = future_price_change * self.shares_held * 0.2 * time_weight
                
                # 🔥 디버깅: 매수 후 미래 가격 변화 보상 (1000 스텝마다)
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(
                        f"[Step {self.current_step}] 매수 후 미래 가격 변화 보상:\n"
                        f"  - 매수 스텝: {self.last_buy_step}\n"
                        f"  - 매수 후 경과 스텝: {steps_since_buy}\n"
                        f"  - 매수 가격: {buy_price:,.0f}원\n"
                        f"  - 현재 가격: {current_price:,.0f}원\n"
                        f"  - 가격 변화율: {future_price_change:.4f}\n"
                        f"  - 시간 가중치: {time_weight:.2f}\n"
                        f"  - 보상: {future_price_reward:.6f}"
                    )

        # 5. 거래 수수료 패널티
        if action > 0:  # 매수나 매도 행동을 했을 때
            transaction_penalty = -self.transaction_fee * 0.01

        # 6. 보유 주식 장려 보상 (새로 추가)
        if self.shares_held > 0:
            holding_encouragement = 0.001 * self.shares_held  # 보유 주식 수에 비례한 보상

        # 최종 보상 계산
        reward = short_term_reward + long_term_reward + holding_reward + future_price_reward + transaction_penalty + holding_encouragement
        self.total_reward += reward

        # TensorBoard 기록
        self.train_step += 1
        if self.writer:
            self.writer.add_scalar("Portfolio Value", new_portfolio_value, self.train_step)
            self.writer.add_scalar("Shares Held", self.shares_held, self.train_step)
            self.writer.add_scalar("Reward/Short-Term", short_term_reward, self.train_step)
            self.writer.add_scalar("Reward/Long-Term", long_term_reward, self.train_step)
            self.writer.add_scalar("Reward/Holding", holding_reward, self.train_step)
            self.writer.add_scalar("Reward/Future-Price", future_price_reward, self.train_step)
            self.writer.add_scalar("Reward/Transaction", transaction_penalty, self.train_step)
            self.writer.add_scalar("Reward/Total", reward, self.train_step)

        # 보상 정규화
        reward = self.normalize_reward(reward, scale=1000)

        # 현재 포트폴리오 가치를 이전 값으로 저장 
        self.previous_portfolio_value = new_portfolio_value

        # 보유 주식 수 히스토리를 저장하는 배열 추가
        if not hasattr(self, "shares_held_history"):
            self.shares_held_history = np.zeros(self.observation_window)

        # 가장 오래된 값을 제거하고, 새로운 보유 주식 수 추가
        self.shares_held_history = np.roll(self.shares_held_history, shift=-1)
        scaled_shares = self.normalize_shares_for_learning(self.shares_held)  # 🔥 스케일된 값으로 업데이트
        self.shares_held_history[-1] = scaled_shares

        # 과거 보유 주식 수 기록을 상태와 함께 결합
        shares_held_feature = self.shares_held_history.reshape(-1, 1)  # (observation_window, 1)
        next_state_with_shares = np.hstack((next_state, shares_held_feature))

        # 🔥 디버깅: 주식 보유량 스케일링 확인 (1000 스텝마다)
        if self.train_step % 1000 == 0:
            log_manager.logger.debug(
                f"[Step {self.current_step}] 주식 보유량 스케일링 확인:\n"
                f"  - 실제 보유 주식 수: {self.shares_held} (정수)\n"
                f"  - 스케일된 보유 주식 수: {scaled_shares:.4f} (학습용)\n"
                # f"  - 스케일링 팩터: {self.shares_scaling_factor}\n"
            )

        # log_manager.logger.debug(f"Step: {self.current_step}, Action: {['Sell', 'Hold', 'Buy'][action]}, Reward: {reward}, Portfolio: {new_portfolio_value}, Shares Held: {self.shares_held}")

        # 입력 state 로그 출력해보기
        # self.feature_names = [
        #                     "D_Close",
        #                     "D_Slope_SMA_5", "D_Slope_SMA_10", "D_Slope_SMA_15", "D_Slope_SMA_20",
        #                     "W_Slope_SMA_5", "W_Slope_SMA_10",
        #                     "M_Slope_SMA_5",
        #                     "보유 주식 수"
        #                 ]

        # # ✅ 마지막 시점 상태 출력 (사람이 읽기 쉽게 각 항목 설명)
        # last_state_row = next_state_with_shares[-1]  # 마지막 timestep의 입력
        # log_msg = f"[Step {self.current_step}] 📥 입력 상태 (가장 최근 시점):\n"
        # for name, value in zip(self.feature_names, last_state_row):
        #     log_msg += f" - {name}: {value:.4f}\n"
        # log_manager.logger.debug(log_msg.strip())

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
