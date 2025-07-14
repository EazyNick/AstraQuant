"""
시각화(그래프 생성) 관련 클래스
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


class Visualizer:
    """그래프 생성 및 시각화 클래스"""
    
    def __init__(self, log_manager):
        self.log_manager = log_manager
        
    def create_comparison_charts(self, portfolio_analyzer, initial_portfolio_value, trading_signals=None):
        """AI 포트폴리오와 Buy-and-Hold 전략 비교 그래프 생성"""
        self.log_manager.logger.info("📊 AI 포트폴리오와 Buy-and-Hold 전략 비교 그래프를 생성합니다...")
        
        self._create_portfolio_comparison_chart(portfolio_analyzer, trading_signals)
        self._create_normalized_comparison_chart(portfolio_analyzer, initial_portfolio_value, trading_signals)
        
    def _create_portfolio_comparison_chart(self, portfolio_analyzer, trading_signals=None):
        """포트폴리오 비교 차트 생성"""
        # 그래프 설정 - 단일 y축 사용
        plt.figure(figsize=(15, 10))
        
        # AI 포트폴리오와 Buy-and-Hold 포트폴리오 비교
        plt.plot(portfolio_analyzer.tracking_dates, portfolio_analyzer.portfolio_values, 'b-', linewidth=2, label='AI Portfolio')
        plt.plot(portfolio_analyzer.tracking_dates, portfolio_analyzer.buy_and_hold_values, 'r-', linewidth=2, label='Buy-and-Hold Strategy')
        
        # 매수/매도 시점 표시
        if trading_signals:
            # 매수 시점을 포트폴리오 가치로 매핑
            buy_portfolio_values = []
            sell_portfolio_values = []
            
            for buy_date in trading_signals['buy_dates']:
                # 해당 날짜의 포트폴리오 가치 찾기
                idx = None
                for i, date in enumerate(portfolio_analyzer.tracking_dates):
                    if date == buy_date:
                        idx = i
                        break
                if idx is not None:
                    buy_portfolio_values.append(portfolio_analyzer.portfolio_values[idx])
                    
            for sell_date in trading_signals['sell_dates']:
                # 해당 날짜의 포트폴리오 가치 찾기
                idx = None
                for i, date in enumerate(portfolio_analyzer.tracking_dates):
                    if date == sell_date:
                        idx = i
                        break
                if idx is not None:
                    sell_portfolio_values.append(portfolio_analyzer.portfolio_values[idx])
            
            # 매수/매도 점 표시
            if trading_signals['buy_dates'] and buy_portfolio_values:
                plt.scatter(trading_signals['buy_dates'], buy_portfolio_values, 
                           color='green', marker='^', s=100, label='AI Buy Signal', zorder=5, alpha=0.8)
                # 매수 시점에서 수직 점선 추가
                for i, buy_date in enumerate(trading_signals['buy_dates']):
                    if i < len(buy_portfolio_values):
                        plt.axvline(x=buy_date, color='green', linestyle='--', alpha=0.6, linewidth=1)
                        
            if trading_signals['sell_dates'] and sell_portfolio_values:
                plt.scatter(trading_signals['sell_dates'], sell_portfolio_values, 
                           color='red', marker='v', s=100, label='AI Sell Signal', zorder=5, alpha=0.8)
                # 매도 시점에서 수직 점선 추가
                for i, sell_date in enumerate(trading_signals['sell_dates']):
                    if i < len(sell_portfolio_values):
                        plt.axvline(x=sell_date, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value (KRW)', fontsize=12)
        plt.title('AI Portfolio vs Buy-and-Hold Strategy Comparison', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # x축 날짜 형식 설정
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
        
        # 그래프 저장
        plt.tight_layout()
        chart_path = "output/ai_vs_buy_hold_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        self.log_manager.logger.info(f"📈 그래프가 저장되었습니다: {chart_path}")
        plt.close()  # 첫 번째 그래프 창 닫기
        
    def _create_normalized_comparison_chart(self, portfolio_analyzer, initial_portfolio_value, trading_signals=None):
        """정규화된 비교 차트 생성"""
        # 정규화된 비교 그래프도 생성
        plt.figure(figsize=(15, 8))
        
        # 정규화를 위해 초기값으로 값들을 나누어 비율로 표시
        portfolio_normalized = np.array(portfolio_analyzer.portfolio_values) / initial_portfolio_value
        buy_and_hold_normalized = np.array(portfolio_analyzer.buy_and_hold_values) / initial_portfolio_value
        
        plt.plot(portfolio_analyzer.tracking_dates, portfolio_normalized, 'b-', linewidth=2, label='AI Portfolio (Normalized)')
        plt.plot(portfolio_analyzer.tracking_dates, buy_and_hold_normalized, 'r-', linewidth=2, label='Buy-and-Hold Strategy (Normalized)')
        
        # 매수/매도 시점 표시 (정규화된 값으로)
        if trading_signals:
            # 매수 시점을 정규화된 포트폴리오 가치로 매핑
            buy_normalized_values = []
            sell_normalized_values = []
            
            for buy_date in trading_signals['buy_dates']:
                # 해당 날짜의 정규화된 포트폴리오 가치 찾기
                idx = None
                for i, date in enumerate(portfolio_analyzer.tracking_dates):
                    if date == buy_date:
                        idx = i
                        break
                if idx is not None:
                    buy_normalized_values.append(portfolio_normalized[idx])
                    
            for sell_date in trading_signals['sell_dates']:
                # 해당 날짜의 정규화된 포트폴리오 가치 찾기
                idx = None
                for i, date in enumerate(portfolio_analyzer.tracking_dates):
                    if date == sell_date:
                        idx = i
                        break
                if idx is not None:
                    sell_normalized_values.append(portfolio_normalized[idx])
            
            # 매수/매도 점 표시
            if trading_signals['buy_dates'] and buy_normalized_values:
                plt.scatter(trading_signals['buy_dates'], buy_normalized_values, 
                           color='green', marker='^', s=100, label='AI Buy Signal', zorder=5, alpha=0.8)
                # 매수 시점에서 수직 점선 추가
                for i, buy_date in enumerate(trading_signals['buy_dates']):
                    if i < len(buy_normalized_values):
                        plt.axvline(x=buy_date, color='green', linestyle='--', alpha=0.6, linewidth=1)
                        
            if trading_signals['sell_dates'] and sell_normalized_values:
                plt.scatter(trading_signals['sell_dates'], sell_normalized_values, 
                           color='red', marker='v', s=100, label='AI Sell Signal', zorder=5, alpha=0.8)
                # 매도 시점에서 수직 점선 추가
                for i, sell_date in enumerate(trading_signals['sell_dates']):
                    if i < len(sell_normalized_values):
                        plt.axvline(x=sell_date, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Performance Ratio (Initial = 1.0)', fontsize=12)
        plt.title('AI Portfolio vs Buy-and-Hold Strategy Performance (Normalized)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # x축 날짜 형식 설정
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        normalized_chart_path = "output/ai_vs_buy_hold_normalized.png"
        plt.savefig(normalized_chart_path, dpi=300, bbox_inches='tight')
        self.log_manager.logger.info(f"📈 정규화 그래프가 저장되었습니다: {normalized_chart_path}")
        
        # 그래프 표시 (옵션)
        plt.show() 