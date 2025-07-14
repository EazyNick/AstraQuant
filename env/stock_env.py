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
        
        # Close 가격을 별도로 보관 (포트폴리오 계산용)
        self.close_prices = stock_data[:, 0]  # Close 가격만 별도 보관
        
        # Close를 제외한 학습용 데이터 생성
        self.stock_data = np.delete(stock_data, 0, axis=1)  # 첫 번째 컬럼(Close) 제거
        
        self.feature_dim = self.stock_data.shape[1] # Close 제외한 feature 개수
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0 # 보유 주식 수
        self.close_price_scale = 10  # 'Close' 값을 원래 가격으로 복원할 때 사용할 스케일
        self.previous_portfolio_value = self.initial_balance
        
        # 주식 보유량 스케일링 개선
        self.shares_scaling_factor = config_manager.get_shares_scaling_factor()  # 설정에서 가져오기
        
        # 액션 공간을 3개로 단순화: 0=관망, 1=전부매수, 2=전부매도
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_window, self.feature_dim), dtype=np.float32)

        # ✅ TensorBoard 추가
        self.train_step = 0  # 학습 스텝 카운트
        self.total_reward = 0  # 최종 보상 추적용 변수
        
        # 매수/매도 추적을 위한 변수들
        self.last_buy_price = None
        self.last_sell_price = None
        self.holding_duration = 0  # 보유 기간
        self.max_holding_duration = 60  # 최대 보유 기간 (매도 압박), 장기 우상향 종목 대상

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
        self.last_buy_step = None  # 매수 시점 추적 초기화
        self.last_sell_step = None  # 매도 시점 추적 초기화
        self.last_sold_shares = 0  # 매도한 주식 수 초기화
        self.last_buy_price = None  # 매수 가격 초기화
        self.last_sell_price = None  # 매도 가격 초기화
        self.holding_duration = 0  # 보유 기간 초기화

        # 🔹 기존 상태 (주가 데이터) + 보유 주식 수 추가
        state = self.stock_data[self.current_step:self.current_step + self.observation_window]
        
        # 주식 보유량을 학습용 스케일로 변환
        scaled_shares = self.normalize_shares_for_learning(self.shares_held)
        shares_held_feature = np.full((self.observation_window, 1), scaled_shares)  # 스케일된 보유 주식수를 feature로 추가
        state_with_shares = np.hstack((state, shares_held_feature))  # 상태 확장
        
        # 디버깅: 보유 주식수 feature 확인
        log_manager.logger.debug(
            f"[Reset] 보유 주식수 feature 확인:\n"
            f"  - 원본 state shape: {state.shape}\n"
            f"  - 보유 주식수: {self.shares_held}주\n"
            f"  - 스케일된 보유 주식수: {scaled_shares:.4f}\n"
            f"  - shares_held_feature shape: {shares_held_feature.shape}\n"
            f"  - 최종 state_with_shares shape: {state_with_shares.shape}\n"
            f"  - 마지막 feature 값들: {state_with_shares[:, -1]}"
        )
        
        return state_with_shares

    def step(self, action):
        """ 액션을 실행하고 새로운 상태, 보상, 종료 여부 반환 """
        reward = 0
        price = self.close_prices[self.current_step] * self.close_price_scale  # 별도 보관된 Close 가격 사용
        if np.isnan(price) or price <= 0:
            log_manager.logger.warning(f"[Step {self.current_step}] 경고: 유효하지 않은 가격 {price}.")
            return None, 0, True  # 가격이 NaN이면 종료

        # 보유 기간 업데이트
        if self.shares_held > 0:
            self.holding_duration += 1

        # 매수/매도 성공 여부 추적 변수
        buy_success = False
        sell_success = False

        if action == 0:
            # 관망 (Hold)
            pass

        elif action == 1:
            # 전부 매수 (Buy All)
            if self.balance > 0:  # 잔고가 있는 경우에만 매수
                # 현재 잔고로 살 수 있는 최대 주식 수 계산
                max_shares_possible = int(self.balance / (price * (1 + self.transaction_fee)))
                if max_shares_possible > 0:
                    cost = max_shares_possible * price * (1 + self.transaction_fee)
                    self.shares_held += max_shares_possible
                    self.balance -= cost
                    self.last_buy_price = price  # 매수 가격 기록
                    self.holding_duration = 0  # 보유 기간 초기화
                    self.last_buy_step = self.current_step
                    self.last_sell_step = None  # 매수 시 매도 시점 초기화
                    self.last_sold_shares = 0  # 매수 시 매도 주식 수 초기화
                    buy_success = True  # 매수 성공 표시
                    
                    if self.train_step % 1000 == 0:
                        portfolio_value = self.balance + (self.shares_held * price)
                        log_manager.logger.debug(
                            f"[Step {self.current_step}] 전부 매수 성공:\n"
                            f"  - 액션: {action} (전부 매수)\n"
                            f"  - 현재 가격: {price:,.0f}원\n"
                            f"  - 매수 주식: {max_shares_possible}주\n"
                            f"  - 필요 비용: {cost:,.0f}원\n"
                            f"  - 매수 전 보유 주식: {self.shares_held - max_shares_possible}주\n"
                            f"  - 매수 후 보유 주식: {self.shares_held}주\n"
                            f"  - 매수 전 잔고: {self.balance + cost:,.0f}원\n"
                            f"  - 매수 후 잔고: {self.balance:,.0f}원\n"
                            f"  - 포트폴리오 밸류: {portfolio_value:,.0f}원\n"
                            f"  - 스케일된 보유 주식: {self.normalize_shares_for_learning(self.shares_held):.4f}"
                        )
                else:
                    reward -= 0.01  # 매수 실패 패널티 (잔고 부족)
                    if self.train_step % 1000 == 0:
                        log_manager.logger.debug(f"[Step {self.current_step}] 전부 매수 실패! 잔고 부족")
            else:
                reward -= 0.05  # 매수 실패 패널티 (잔고 없음)
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(f"[Step {self.current_step}] 전부 매수 실패! 잔고 없음")

        elif action == 2:
            # 전부 매도 (Sell All)
            if self.shares_held > 0:  # 보유 주식이 있는 경우에만 매도
                revenue = self.shares_held * price * (1 - self.transaction_fee)
                self.balance += revenue
                shares_sold = self.shares_held
                self.shares_held = 0  # 전부 매도
                self.last_buy_step = None  # 매도 시 매수 시점 초기화
                self.last_sell_step = self.current_step  # 매도 시점 기록
                self.last_sold_shares = shares_sold  # 매도한 주식 수 기록
                self.last_sell_price = price  # 매도 가격 기록
                self.holding_duration = 0  # 보유 기간 초기화
                sell_success = True  # 매도 성공 표시
                
                if self.train_step % 1000 == 0:
                    portfolio_value = self.balance + (self.shares_held * price)
                    log_manager.logger.debug(
                        f"[Step {self.current_step}] 전부 매도 성공:\n"
                        f"  - 액션: {action} (전부 매도)\n"
                        f"  - 매도 주식: {shares_sold}주\n"
                        f"  - 매도 수익: {revenue:,.0f}원\n"
                        f"  - 보유 주식: {self.shares_held}주\n"
                        f"  - 잔고: {self.balance:,.0f}원\n"
                        f"  - 포트폴리오 밸류: {portfolio_value:,.0f}원\n"
                        f"  - 스케일된 보유 주식: {self.normalize_shares_for_learning(self.shares_held):.4f}"
                    )
            else:
                reward -= 0.01  # 매도 실패 패널티 (보유 주식 없음)
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(f"[Step {self.current_step}] 전부 매도 실패! 보유 주식 없음")

        # 새로운 포트폴리오 가치 계산 (현재 스텝의 가격 사용)
        new_portfolio_value = self.balance + (self.shares_held * price)

        # 1. 단기 수익률 보상
        short_term_reward = 0
        long_term_reward = 0
        holding_reward = 0
        future_price_reward = 0
        sell_price_reward = 0
        transaction_penalty = 0
        profit_taking_reward = 0  # 수익 실현 보상
        loss_cutting_reward = 0   # 손실 절단 보상
        holding_pressure = 0      # 보유 압박 (장기 보유 시 패널티)

        # 포트폴리오 가치 변화율을 보상으로 설정 (수익률 기반 보상), 단기 수익률 보상
        if self.previous_portfolio_value > 0:
            short_term_reward = ((new_portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value) * 12

        # 2. 장기 수익률 보상 (현재 가치 대비 초기 가치)
        long_term_reward = ((new_portfolio_value - self.initial_balance) / self.initial_balance) * 10

        # 3. 보유 주식 가격 변화 보상
        if self.shares_held > 0 and self.current_step > 1:  # 첫 번째 스텝이 아닌 경우에만 계산
            # -1을 해야 이전 스텝과 현재 스텝을 비교
            prev_price = self.close_prices[self.current_step - 1] * self.close_price_scale  # 별도 보관된 Close 가격 사용
            price_change = (price - prev_price) / prev_price
            holding_reward = price_change * self.shares_held * 0.1

        # 4. 매수 후 미래 가격 변화 보상 (매수 성공 시에만 적용)
        if buy_success and self.shares_held > 0 and self.last_buy_step is not None:
            steps_since_buy = self.current_step - self.last_buy_step
            
            # 매수 후 1-5 스텝 동안의 가격 변화를 고려
            if 1 <= steps_since_buy <= 5:
                buy_price = self.close_prices[self.last_buy_step] * self.close_price_scale
                current_price = price  # 현재 스텝의 가격
                future_price_change = (current_price - buy_price) / buy_price
                
                # 매수 후 가격 상승 시 양의 보상, 하락 시 음의 보상
                # 시간이 지날수록 보상 가중치 감소 (즉시 반응을 더 중요하게)
                time_weight = 1.0 / steps_since_buy  # 1스텝 후: 1.0, 2스텝 후: 0.5, 3스텝 후: 0.33...
                future_price_reward = future_price_change * self.shares_held * 0.2 * time_weight
                
                # 디버깅: 매수 후 미래 가격 변화 보상 (1000 스텝마다)
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(
                        f"[Step {self.current_step}] 매수 후 미래 가격 변화 보상:\n"
                        f"  - 매수 성공: {buy_success}\n"
                        f"  - 매수 스텝: {self.last_buy_step}\n"
                        f"  - 매수 후 경과 스텝: {steps_since_buy}\n"
                        f"  - 매수 가격: {buy_price:,.0f}원\n"
                        f"  - 현재 가격: {current_price:,.0f}원\n"
                        f"  - 가격 변화율: {future_price_change:.4f}\n"
                        f"  - 시간 가중치: {time_weight:.2f}\n"
                        f"  - 보상: {future_price_reward:.6f}"
                    )

        # 5. 매도 후 미래 가격 변화 보상 (매도 성공 시에만 적용)
        if sell_success and self.shares_held == 0 and self.last_sell_step is not None:
            steps_since_sell = self.current_step - self.last_sell_step
            
            # 매도 후 1-5 스텝 동안의 가격 변화를 고려
            if 1 <= steps_since_sell <= 5:
                sell_price = self.close_prices[self.last_sell_step] * self.close_price_scale
                current_price = price  # 현재 스텝의 가격
                price_change_since_sell = (current_price - sell_price) / sell_price
                
                # 매도 후 가격 상승 시 음의 보상(패널티), 하락 시 양의 보상
                # 매도를 너무 일찍 했으면 패널티, 적절한 타이밍이면 보상
                time_weight = 1.0 / steps_since_sell  # 1스텝 후: 1.0, 2스텝 후: 0.5, 3스텝 후: 0.33...
                sell_price_reward = -price_change_since_sell * self.last_sold_shares * 0.2 * time_weight
                
                # 디버깅: 매도 후 미래 가격 변화 보상 (1000 스텝마다)
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(
                        f"[Step {self.current_step}] 매도 후 미래 가격 변화 보상:\n"
                        f"  - 매도 성공: {sell_success}\n"
                        f"  - 매도 스텝: {self.last_sell_step}\n"
                        f"  - 매도 후 경과 스텝: {steps_since_sell}\n"
                        f"  - 매도 가격: {sell_price:,.0f}원\n"
                        f"  - 현재 가격: {current_price:,.0f}원\n"
                        f"  - 가격 변화율: {price_change_since_sell:.4f}\n"
                        f"  - 시간 가중치: {time_weight:.2f}\n"
                        f"  - 보상: {sell_price_reward:.6f}"
                    )

        # 6. 거래 수수료 패널티 (매수/매도 성공 시에만 적용)
        if (buy_success or sell_success):  # 매수나 매도가 성공했을 때만
            transaction_penalty = -self.transaction_fee * 0.01

        # 7. 수익 실현 보상 (매도 성공 시 수익이 있을 때)
        if sell_success and self.last_buy_price is not None:  # 매도 성공이고 매수 가격이 기록되어 있을 때
            profit_rate = (price - self.last_buy_price) / self.last_buy_price
            if profit_rate > 0.02:  # 2% 이상 수익 시
                profit_taking_reward = profit_rate * 5.0  # 수익률에 비례한 보상
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(f"[Step {self.current_step}] 수익 실현 보상: {profit_taking_reward:.6f} (수익률: {profit_rate:.4f})")

        # 8. 손실 절단 보상 (매도 성공 시 손실이 있을 때)
        if sell_success and self.last_buy_price is not None:  # 매도 성공이고 매수 가격이 기록되어 있을 때
            loss_rate = (self.last_buy_price - price) / self.last_buy_price
            if loss_rate > 0.01:  # 1% 이상 손실 시
                loss_cutting_reward = loss_rate * 2.0  # 손실 절단에 대한 보상
                if self.train_step % 1000 == 0:
                    log_manager.logger.debug(f"[Step {self.current_step}] 손실 절단 보상: {loss_cutting_reward:.6f} (손실률: {loss_rate:.4f})")

        # 9. 보유 압박 (장기 보유 시 패널티) - 매도 유도
        if self.shares_held > 0 and self.holding_duration > self.max_holding_duration:
            holding_pressure = -0.01 * (self.holding_duration - self.max_holding_duration)  # 보유 기간이 길수록 패널티
            if self.train_step % 1000 == 0:
                log_manager.logger.debug(f"[Step {self.current_step}] 보유 압박 패널티: {holding_pressure:.6f} (보유 기간: {self.holding_duration})")

        # 최종 보상 계산
        reward = (short_term_reward + long_term_reward + holding_reward + 
                 future_price_reward + sell_price_reward + transaction_penalty + 
                 profit_taking_reward + loss_cutting_reward + holding_pressure)
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
            self.writer.add_scalar("Reward/Sell-Price", sell_price_reward, self.train_step)
            self.writer.add_scalar("Reward/Transaction", transaction_penalty, self.train_step)
            self.writer.add_scalar("Reward/Profit-Taking", profit_taking_reward, self.train_step)
            self.writer.add_scalar("Reward/Loss-Cutting", loss_cutting_reward, self.train_step)
            self.writer.add_scalar("Reward/Holding-Pressure", holding_pressure, self.train_step)
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
        scaled_shares = self.normalize_shares_for_learning(self.shares_held)  # 스케일된 값으로 업데이트
        self.shares_held_history[-1] = scaled_shares

        # 과거 보유 주식 수 기록을 상태와 함께 결합
        shares_held_feature = self.shares_held_history.reshape(-1, 1)  # (observation_window, 1)
        
        # current_step 증가를 먼저 실행
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - self.observation_window
        
        # 다음 스텝의 상태 계산 (current_step 증가 후)
        next_state = self.stock_data[self.current_step:self.current_step + self.observation_window]
        next_state_with_shares = np.hstack((next_state, shares_held_feature))

        # 디버깅: 주식 보유량 스케일링 확인 (1000 스텝마다)
        if self.train_step % 1000 == 0:
            log_manager.logger.debug(
                f"[Step {self.current_step - 1}] 주식 보유량 스케일링 확인:\n"
                f"  - 실제 보유 주식 수: {self.shares_held} (정수)\n"
                f"  - 스케일된 보유 주식 수: {scaled_shares:.4f} (학습용)\n"
                f"  - shares_held_history: {self.shares_held_history}\n"
                f"  - shares_held_feature shape: {shares_held_feature.shape}\n"
                f"  - next_state shape: {next_state.shape}\n"
                f"  - next_state_with_shares shape: {next_state_with_shares.shape}\n"
                f"  - 마지막 feature 값들: {next_state_with_shares[:, -1]}"
            )

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
