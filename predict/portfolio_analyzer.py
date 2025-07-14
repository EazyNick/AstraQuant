"""
포트폴리오 분석 관련 클래스
"""

import numpy as np
import pandas as pd


class PortfolioAnalyzer:
    """포트폴리오 성과 분석 클래스"""
    
    def __init__(self, initial_balance, transaction_fee, log_manager):
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.log_manager = log_manager
        
        # 포트폴리오 추적 변수들
        self.balance = initial_balance
        self.holding = 0
        self.portfolio_values = []
        self.close_prices = []
        self.buy_and_hold_values = []
        self.tracking_dates = []
        
        # Buy-and-Hold 전략 변수들
        self.buy_and_hold_shares = 0
        self.buy_and_hold_remaining_cash = 0
        
    def initialize_buy_and_hold(self, initial_price):
        """Buy-and-Hold 전략 초기화"""
        self.buy_and_hold_shares = int(self.initial_balance / (initial_price * (1 + self.transaction_fee)))
        self.buy_and_hold_remaining_cash = self.initial_balance - (self.buy_and_hold_shares * initial_price * (1 + self.transaction_fee))
        
    def execute_action(self, action, current_price):
        """매매 액션 실행"""
        if action == 1:  # 전량 매수 (Buy All)
            if self.balance > 0:  # 잔고가 있는 경우에만 매수
                # 현재 잔고로 살 수 있는 최대 주식 수 계산
                max_shares_possible = int(self.balance / (current_price * (1 + self.transaction_fee)))
                if max_shares_possible > 0:
                    cost = max_shares_possible * current_price * (1 + self.transaction_fee)
                    self.holding += max_shares_possible
                    self.balance -= cost
                    print(f"전량 매수: {max_shares_possible}주 → 총 보유: {self.holding}주, 잔고: {self.balance:,.0f}원")
        elif action == 2:  # 전량 매도 (Sell All)
            if self.holding > 0:  # 보유 주식이 있는 경우에만 매도
                revenue = self.holding * current_price * (1 - self.transaction_fee)
                shares_sold = self.holding
                self.holding = 0  # 전량 매도
                self.balance += revenue
                print(f"전량 매도: {shares_sold}주 → 총 보유: {self.holding}주, 잔고: {self.balance:,.0f}원")
                
    def update_tracking(self, current_price, date):
        """포트폴리오 추적 데이터 업데이트"""
        # 현재 포트폴리오 밸류 계산 및 저장
        current_portfolio_value = self.balance + (self.holding * current_price)
        self.portfolio_values.append(current_portfolio_value)
        self.close_prices.append(current_price)
        
        # Buy-and-Hold 전략 포트폴리오 가치 계산
        buy_and_hold_portfolio_value = self.buy_and_hold_remaining_cash + (self.buy_and_hold_shares * current_price)
        self.buy_and_hold_values.append(buy_and_hold_portfolio_value)
        
        self.tracking_dates.append(pd.to_datetime(date))
        
    def calculate_final_performance(self, final_price, initial_price):
        """최종 성과 계산 및 로그 출력"""
        # 최종 포트폴리오 밸류 계산
        final_portfolio_value = self.balance + (self.holding * final_price)
        initial_portfolio_value = self.initial_balance
        
        # 수익률 계산
        total_return = final_portfolio_value - initial_portfolio_value
        return_rate = (total_return / initial_portfolio_value) * 100
        
        self.log_manager.logger.info(f"💰 최종 포트폴리오 정보:")
        self.log_manager.logger.info(f"   - 최종 잔고: {self.balance:,.0f}원")
        self.log_manager.logger.info(f"   - 최종 보유 주식: {self.holding}주")
        self.log_manager.logger.info(f"   - 최종 주가: {final_price:,.0f}원")
        self.log_manager.logger.info(f"   - 최종 포트폴리오 밸류: {final_portfolio_value:,.0f}원")
        
        self.log_manager.logger.info(f"📈 투자 성과 분석:")
        self.log_manager.logger.info(f"   - 총 수익/손실: {total_return:,.0f}원")
        self.log_manager.logger.info(f"   - 수익률: {return_rate:.2f}%")
        
        # 벤치마크 비교 (단순 보유 전략)
        benchmark_return = ((final_price - initial_price) / initial_price) * 100
        excess_return = return_rate - benchmark_return
        
        self.log_manager.logger.info(f"📊 벤치마크 비교 (단순 보유 전략):")
        self.log_manager.logger.info(f"   - 벤치마크 수익률: {benchmark_return:.2f}%")
        self.log_manager.logger.info(f"   - 초과 수익률: {excess_return:.2f}%")
        
        if excess_return > 0:
            self.log_manager.logger.info(f"🎉 AI 모델이 단순 보유 전략보다 {excess_return:.2f}%p 더 좋은 성과를 보였습니다!")
        else:
            self.log_manager.logger.info(f"😞 AI 모델이 단순 보유 전략보다 {abs(excess_return):.2f}%p 낮은 성과를 보였습니다.")

        # Buy-and-Hold 전략 성과 계산
        buy_and_hold_final_value = self.buy_and_hold_values[-1]
        buy_and_hold_return = buy_and_hold_final_value - initial_portfolio_value
        buy_and_hold_return_rate = (buy_and_hold_return / initial_portfolio_value) * 100
        
        self.log_manager.logger.info(f"📊 Buy-and-Hold 전략 성과:")
        self.log_manager.logger.info(f"   - 매수한 주식 수: {self.buy_and_hold_shares}주")
        self.log_manager.logger.info(f"   - 남은 현금: {self.buy_and_hold_remaining_cash:,.0f}원")
        self.log_manager.logger.info(f"   - 최종 포트폴리오 가치: {buy_and_hold_final_value:,.0f}원")
        self.log_manager.logger.info(f"   - 수익률: {buy_and_hold_return_rate:.2f}%")
        
        # AI vs Buy-and-Hold 비교
        ai_vs_buy_hold_excess = return_rate - buy_and_hold_return_rate
        self.log_manager.logger.info(f"🤖 AI vs Buy-and-Hold 비교:")
        self.log_manager.logger.info(f"   - AI 수익률: {return_rate:.2f}%")
        self.log_manager.logger.info(f"   - Buy-and-Hold 수익률: {buy_and_hold_return_rate:.2f}%")
        self.log_manager.logger.info(f"   - AI 초과 수익률: {ai_vs_buy_hold_excess:.2f}%p")
        
        if ai_vs_buy_hold_excess > 0:
            self.log_manager.logger.info(f"🎉 AI 모델이 Buy-and-Hold 전략보다 {ai_vs_buy_hold_excess:.2f}%p 더 좋은 성과를 보였습니다!")
        else:
            self.log_manager.logger.info(f"😞 AI 모델이 Buy-and-Hold 전략보다 {abs(ai_vs_buy_hold_excess):.2f}%p 낮은 성과를 보였습니다.")
            
        return {
            'return_rate': return_rate,
            'final_portfolio_value': final_portfolio_value,
            'initial_portfolio_value': initial_portfolio_value
        } 