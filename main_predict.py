import torch
import numpy as np
import os
import argparse
import pandas as pd
from datetime import datetime

# AstraQuant 모듈들
from models.actor_network import ActorNetwork
from data.data_loader import load_stock_data
from predict import (
    load_model, predict_action, get_prediction_by_date, 
    PortfolioAnalyzer, Visualizer, ResultSaver
)

try:
    from logs import log_manager
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")


def main():
    """메인 실행 함수"""
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'output', 'ppo_stock_trader_episode_296.pth'), 
                       help='사용할 모델 가중치 파일 (.pth) 경로')
    parser.add_argument('--test_data', type=str, 
                       default='data/csv/sp500_test_data.csv', 
                       help='테스트 데이터 (.csv) 파일 경로')
    args = parser.parse_args()

    # 설정 로드
    device = torch.device(config_manager.get_device())
    balance = config_manager.get_initial_balance()
    transaction_fee = config_manager.get_transaction_fee()
    observation_window = config_manager.get_observation_window()

    # 데이터 로드
    stock_data, input_dim = load_stock_data(args.test_data)
    print(f"📊 로드된 데이터 형태: {stock_data.shape}")
    print(f"📐 데이터 입력 차원: {input_dim}")
    
    # 모델 로드
    actor_model = load_model(args.model_path, ActorNetwork, input_dim, device)
    if actor_model is None:
        log_manager.logger.error("❌ 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)

    # 데이터프레임 로드 및 날짜 처리
    df = pd.read_csv(args.test_data or 'data/cursor_csv/AMZN_test_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    dates = df['Date'].values

    # 데이터 검증
    if stock_data.shape[0] < observation_window:
        raise ValueError(f"❌ 테스트 데이터가 너무 적습니다! (필요: {observation_window}, 제공됨: {stock_data.shape[0]})")

    if len(dates) > stock_data.shape[0]:
        dates = dates[-stock_data.shape[0]:]

    # 포트폴리오 분석기 초기화
    portfolio_analyzer = PortfolioAnalyzer(balance, transaction_fee, log_manager)
    
    # 초기 설정
    initial_price = stock_data[observation_window, 0] * 10
    initial_portfolio_value = balance
    portfolio_analyzer.initialize_buy_and_hold(initial_price)
    
    log_manager.logger.info(f"💰 초기 포트폴리오 정보:")
    log_manager.logger.info(f"   - 초기 잔고: {balance:,.0f}원")
    log_manager.logger.info(f"   - 초기 주가: {initial_price:,.0f}원")
    log_manager.logger.info(f"   - 초기 포트폴리오 밸류: {initial_portfolio_value:,.0f}원")

    # 예측 실행
    predictions, probs_list, trading_signals = run_predictions(
        actor_model, stock_data, dates, observation_window, 
        portfolio_analyzer, device
    )
    
    # 성과 분석
    final_price = stock_data[-1, 0] * 10
    initial_price = stock_data[observation_window, 0] * 10
    performance_results = portfolio_analyzer.calculate_final_performance(final_price, initial_price)
    
    # 매매 시그널 요약 로그
    log_manager.logger.info(f"📊 AI 매매 시그널 요약:")
    log_manager.logger.info(f"   - 총 매수 시점: {len(trading_signals['buy_dates'])}회")
    log_manager.logger.info(f"   - 총 매도 시점: {len(trading_signals['sell_dates'])}회")
    if trading_signals['buy_dates']:
        avg_buy_price = sum(trading_signals['buy_prices']) / len(trading_signals['buy_prices'])
        log_manager.logger.info(f"   - 평균 매수 가격: {avg_buy_price:,.0f}원")
    if trading_signals['sell_dates']:
        avg_sell_price = sum(trading_signals['sell_prices']) / len(trading_signals['sell_prices'])
        log_manager.logger.info(f"   - 평균 매도 가격: {avg_sell_price:,.0f}원")
    
    # 시각화
    visualizer = Visualizer(log_manager)
    visualizer.create_comparison_charts(portfolio_analyzer, initial_portfolio_value, trading_signals)
    
    # 결과 저장 및 출력
    result_saver = ResultSaver(log_manager)
    result_df = pd.DataFrame(predictions, columns=["날짜", "예측 매매 결정", "확률(%)"])
    
    result_saver.log_prediction_summary(result_df, probs_list[-1])
    # 마지막 액션 인덱스 찾기
    last_action_str = predictions[-1][1]
    if "관망" in last_action_str:
        last_action_idx = 0
    elif "매수" in last_action_str:
        last_action_idx = 1
    else:  # 매도
        last_action_idx = 2
    
    result_saver.log_final_predictions(predictions, probs_list[-1], last_action_idx, df)
    result_saver.save_prediction_results(predictions, probs_list)
    
    # 특정 날짜 예측 조회 예시
    target_date = "2024-07-12"
    action_str, prob = get_prediction_by_date(result_df, target_date)
    if action_str is not None:
        log_manager.logger.info(f"📅 [{target_date}] 예측 결과: {action_str} (확률: {prob:.2f}%)")


def run_predictions(actor_model, stock_data, dates, observation_window, portfolio_analyzer, device):
    """예측 실행 함수"""
    # 액션 딕셔너리 정의
    action_dict = {
        0: "관망(Hold)",
        1: "전부매수(Buy All)", 
        2: "전부매도(Sell All)"
    }

    predictions = []
    probs_list = []
    trading_signals = {
        'buy_dates': [],
        'buy_prices': [], 
        'sell_dates': [],
        'sell_prices': []
    }
    
    for i in range(observation_window, stock_data.shape[0]):
        # 상태 데이터 준비
        state = stock_data[i - observation_window:i]
        date = dates[i]
        
        try:
            # 예측 수행
            action, probs = predict_action(actor_model, state, device)
            predictions.append([date, action_dict[action], probs[0][action]])
            probs_list.append(probs[0])
            
            # 현재 주가
            current_price = stock_data[i, 0] * 10
            
            # 매매 실행 전 보유량 저장
            prev_holding = portfolio_analyzer.holding
            
            # 매매 실행
            portfolio_analyzer.execute_action(action, current_price)
            
            # 매매 신호 기록 (실제로 거래가 발생한 경우만)
            if action == 1 and portfolio_analyzer.holding > prev_holding:  # 매수 발생
                trading_signals['buy_dates'].append(pd.to_datetime(date))
                trading_signals['buy_prices'].append(current_price)
            elif action == 2 and portfolio_analyzer.holding < prev_holding:  # 매도 발생
                trading_signals['sell_dates'].append(pd.to_datetime(date))
                trading_signals['sell_prices'].append(current_price)
            
            # 추적 데이터 업데이트
            portfolio_analyzer.update_tracking(current_price, date)
            
        except Exception as e:
            log_manager.logger.error(f"❌ 예측 중 오류 발생: {e}")
            continue
    
    return predictions, probs_list, trading_signals


if __name__ == "__main__":
    main()
    
    # 예시 명령어
    # python main_predict.py --model_path output/ppo_stock_trader_episode_350.pth --test_data data/csv/005930.KS_combined_train_data.csv
    # python main_predict.py --model_path output/ppo_stock_trader_episode_4300.pth --test_data data/csv/005930.KS_combined_test_data.csv