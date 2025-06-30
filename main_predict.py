import torch
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from models.actor_network import ActorNetwork
from data.data_loader import load_stock_data

try:
    from logs import log_manager
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")

# ✅ 저장된 모델 가중치를 불러오는 함수
def load_model(model_path, model_class, input_dim, device="cpu"):
    """
    저장된 모델 가중치를 불러오는 함수

    Args:
        model_path (str): 모델 가중치 파일 경로
        model_class (torch.nn.Module): 모델 클래스 (StockTransformer 등)
        input_dim (int): 모델 입력 차원
        device (str, optional): 사용할 디바이스. 기본값: "cpu"

    Returns:
        model (torch.nn.Module): 불러온 모델 (None 반환 시 로드 실패)
    """
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return None

    try:
        # ✅ 새로운 모델 객체를 먼저 생성한 후, 가중치를 불러옴
        model = model_class(input_dim=input_dim).to(device)

        # ✅ 가중치 로드 (PyTorch 2.6 이후 버전 대응)
        state_dict = torch.load(model_path, map_location=device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"📐 모델 입력 차원: {input_dim}")
        if missing:
            print(f"⚠️ 누락된 가중치: {missing}")
        if unexpected:
            print(f"⚠️ 예상치 못한 키: {unexpected}")

        # ✅ 모델 평가 모드 설정
        model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")

        return model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None

# 확률 값 변환 함수 (0~100% 범위로 변환 및 최소값 보장)
def format_probs(probs):
    normalized_probs = probs * 100 # 확률을 0~100 범위로 변환
    formatted_probs = np.maximum(normalized_probs, 0.01) # 최소값 0.01% 보장
    return np.round(formatted_probs, 2) # 소수점 2자리까지 변환하여 출력

# 매매 결정 함수
def predict_action(model, state, device):
    if model is None:
        raise ValueError("모델이 로드되지 않았습니다. 모델을 먼저 로드해주세요.")
    
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq_len, feature_dim) 변환
    
    with torch.no_grad():
        logits = model(state)
        probs = torch.softmax(logits, dim=-1) # 확률 계산
        action = torch.argmax(probs, dim=-1).item() # 가장 높은 확률의 액션 선택
        return action, format_probs(probs.cpu().detach().numpy()) # 액션과 확률 반환

def get_prediction_by_date(result_df, target_date: str):
    """
    예측 결과에서 특정 날짜에 해당하는 액션과 확률을 반환

    Args:
        result_df (pd.DataFrame): 예측 결과 데이터프레임
        target_date (str): 조회할 날짜 (예: "2023-12-01")

    Returns:
        tuple: (예측 매매 결정(str), 확률(float)) 또는 (None, None) ← 해당 날짜 없을 경우
    """
    row = result_df[result_df["날짜"] == target_date]
    if row.empty:
        print(f"❌ 날짜 '{target_date}'에 대한 예측 결과가 없습니다.")
        return None, None
    action_str = row.iloc[0]["예측 매매 결정"]
    prob = row.iloc[0]["확률(%)"]
    if isinstance(prob, np.ndarray):
        prob = prob.item()
    else:
        prob = float(prob)
    return action_str, prob

if __name__ == "__main__":
    import pandas as pd
    # ✅ 설정 가져오기
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(__file__), 'output', 'ppo_stock_trader_episode_296.pth'), help='사용할 모델 가중치 파일 (.pth) 경로 (기본값: ppo_stock_trader_episode_288.pth)')
    parser.add_argument('--test_data', type=str, default='data/csv/sp500_test_data.csv', help='테스트 데이터 (.csv) 파일 경로 (기본값: data/csv/sp500_test_data.csv)')
    args = parser.parse_args()

    device = torch.device(config_manager.get_device())
    balance = config_manager.get_initial_balance()
    transaction_fee = config_manager.get_transaction_fee()
    # ✅ 초기 보유 수량
    holding = 0
    
    # ✅ 초기 포트폴리오 밸류 기록
    initial_balance = balance
    initial_holding = holding

    # ✅ 저장된 모델 로드 
    stock_data, input_dim = load_stock_data(args.test_data)
    print(f"📊 로드된 데이터 형태: {stock_data.shape}")
    print(f"📐 데이터 입력 차원: {input_dim}")
    
    # ✅ 환경에서 Close 제거하고 shares held 추가하므로 input_dim 그대로 사용
    actor_model = load_model(args.model_path, ActorNetwork, input_dim, device)
    
    if actor_model is None:
        log_manager.logger.error("❌ 모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)

    # df = pd.read_csv(args.test_data or 'data/csv/AMZN_test_data.csv')
    df = pd.read_csv(args.test_data or 'data/cursor_csv/AMZN_test_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])  # 날짜 형식 변환
    df = df.sort_values('Date').reset_index(drop=True)
    dates = df['Date'].values  # 다시 정렬된 날짜로 업데이트

    log_manager.logger.info(f"📅 전체 테스트 데이터 마지막 날짜: {dates[-1]}")

    # ✅ 마지막 observation_window 만큼의 데이터 가져오기
    observation_window = config_manager.get_observation_window()
    if stock_data.shape[0] < observation_window:
        raise ValueError(f"❌ 테스트 데이터가 너무 적습니다! (필요: {observation_window}, 제공됨: {stock_data.shape[0]})")

    # ✅ 초기 주가 기록 (첫 번째 예측 시점의 주가)
    initial_price = stock_data[observation_window, 0] * 10
    initial_portfolio_value = initial_balance + (initial_holding * initial_price)
    
    log_manager.logger.info(f"💰 초기 포트폴리오 정보:")
    log_manager.logger.info(f"   - 초기 잔고: {initial_balance:,.0f}원")
    log_manager.logger.info(f"   - 초기 보유 주식: {initial_holding}주")
    log_manager.logger.info(f"   - 초기 주가: {initial_price:,.0f}원")
    log_manager.logger.info(f"   - 초기 포트폴리오 밸류: {initial_portfolio_value:,.0f}원")

    # ✅ 전체 데이터에 대한 예측 수행
    action_dict = {}
    # 🔥 3개 액션으로 단순화: 0=관망, 1=전부매수, 2=전부매도
    action_dict[0] = "관망(Hold)"
    action_dict[1] = "전부매수(Buy All)"
    action_dict[2] = "전부매도(Sell All)"

    predictions = []
    probs_list = []  # ✅ 확률 리스트 저장용
    
    # ✅ 포트폴리오 밸류와 Buy-and-Hold 전략 추적을 위한 리스트 추가
    portfolio_values = []
    close_prices = []
    buy_and_hold_values = []
    tracking_dates = []
    
    # ✅ Buy-and-Hold 전략 초기 설정 (초기 자금으로 최대한 매수)
    buy_and_hold_shares = int(initial_balance / (initial_price * (1 + transaction_fee)))
    buy_and_hold_remaining_cash = initial_balance - (buy_and_hold_shares * initial_price * (1 + transaction_fee))

    # ✅ stock_data 크기만큼 앞에서 자르기
    if len(dates) > stock_data.shape[0]:
        dates = dates[-stock_data.shape[0]:]  # 뒤쪽 기준으로 자르기

    for i in range(observation_window, stock_data.shape[0]):
        # ✅ 환경을 통해 상태를 받아야 함 (shares held 포함)
        # 예측에서는 환경을 사용하지 않으므로, 원본 데이터만 사용
        state = stock_data[i - observation_window:i] # 관찰 윈도우 데이터 추출
        date = dates[i] # 해당 날짜 가져오기
        
        try:
            action, probs = predict_action(actor_model, state, device)
            predictions.append([date, action_dict[action], probs[0][action]])
            current_price = stock_data[i, 0] * 10 
            probs_list.append(probs[0])  # ✅ 확률 분포 저장
        except Exception as e:
            log_manager.logger.error(f"❌ 예측 중 오류 발생: {e}")
            log_manager.logger.error(f"   - 입력 상태 shape: {state.shape}")
            log_manager.logger.error(f"   - 모델 입력 차원: {input_dim}")
            continue

        # ✅ 보유 수량 업데이트 - 환경과 동일한 전량 매수/매도 로직
        if action == 1:  # 전량 매수 (Buy All)
            if balance > 0:  # 잔고가 있는 경우에만 매수
                # 현재 잔고로 살 수 있는 최대 주식 수 계산
                max_shares_possible = int(balance / (current_price * (1 + transaction_fee)))
                if max_shares_possible > 0:
                    cost = max_shares_possible * current_price * (1 + transaction_fee)
                    holding += max_shares_possible
                    balance -= cost
                    print(f"전량 매수: {max_shares_possible}주 → 총 보유: {holding}주, 잔고: {balance:,.0f}원")
        elif action == 2:  # 전량 매도 (Sell All)
            if holding > 0:  # 보유 주식이 있는 경우에만 매도
                revenue = holding * current_price * (1 - transaction_fee)
                shares_sold = holding
                holding = 0  # 전량 매도
                balance += revenue
                print(f"전량 매도: {shares_sold}주 → 총 보유: {holding}주, 잔고: {balance:,.0f}원")
        # 관망이나 실패한 거래는 로그 출력하지 않음
        
        # ✅ 현재 포트폴리오 밸류 계산 및 저장
        current_portfolio_value = balance + (holding * current_price)
        portfolio_values.append(current_portfolio_value)
        close_prices.append(current_price)
        
        # ✅ Buy-and-Hold 전략 포트폴리오 가치 계산
        buy_and_hold_portfolio_value = buy_and_hold_remaining_cash + (buy_and_hold_shares * current_price)
        buy_and_hold_values.append(buy_and_hold_portfolio_value)
        
        tracking_dates.append(pd.to_datetime(date))

    # ✅ 최종 포트폴리오 밸류 계산
    final_price = stock_data[-1, 0] * 10  # 마지막 주가
    final_portfolio_value = balance + (holding * final_price)
    
    # ✅ 수익률 계산
    total_return = final_portfolio_value - initial_portfolio_value
    return_rate = (total_return / initial_portfolio_value) * 100
    
    log_manager.logger.info(f"💰 최종 포트폴리오 정보:")
    log_manager.logger.info(f"   - 최종 잔고: {balance:,.0f}원")
    log_manager.logger.info(f"   - 최종 보유 주식: {holding}주")
    log_manager.logger.info(f"   - 최종 주가: {final_price:,.0f}원")
    log_manager.logger.info(f"   - 최종 포트폴리오 밸류: {final_portfolio_value:,.0f}원")
    
    log_manager.logger.info(f"📈 투자 성과 분석:")
    log_manager.logger.info(f"   - 총 수익/손실: {total_return:,.0f}원")
    log_manager.logger.info(f"   - 수익률: {return_rate:.2f}%")
    
    # ✅ 벤치마크 비교 (단순 보유 전략)
    benchmark_return = ((final_price - initial_price) / initial_price) * 100
    excess_return = return_rate - benchmark_return
    
    log_manager.logger.info(f"📊 벤치마크 비교 (단순 보유 전략):")
    log_manager.logger.info(f"   - 벤치마크 수익률: {benchmark_return:.2f}%")
    log_manager.logger.info(f"   - 초과 수익률: {excess_return:.2f}%")
    
    if excess_return > 0:
        log_manager.logger.info(f"🎉 AI 모델이 단순 보유 전략보다 {excess_return:.2f}%p 더 좋은 성과를 보였습니다!")
    else:
        log_manager.logger.info(f"😞 AI 모델이 단순 보유 전략보다 {abs(excess_return):.2f}%p 낮은 성과를 보였습니다.")

    # ✅ Buy-and-Hold 전략 성과 계산
    buy_and_hold_final_value = buy_and_hold_values[-1]
    buy_and_hold_return = buy_and_hold_final_value - initial_portfolio_value
    buy_and_hold_return_rate = (buy_and_hold_return / initial_portfolio_value) * 100
    
    log_manager.logger.info(f"📊 Buy-and-Hold 전략 성과:")
    log_manager.logger.info(f"   - 매수한 주식 수: {buy_and_hold_shares}주")
    log_manager.logger.info(f"   - 남은 현금: {buy_and_hold_remaining_cash:,.0f}원")
    log_manager.logger.info(f"   - 최종 포트폴리오 가치: {buy_and_hold_final_value:,.0f}원")
    log_manager.logger.info(f"   - 수익률: {buy_and_hold_return_rate:.2f}%")
    
    # ✅ AI vs Buy-and-Hold 비교
    ai_vs_buy_hold_excess = return_rate - buy_and_hold_return_rate
    log_manager.logger.info(f"🤖 AI vs Buy-and-Hold 비교:")
    log_manager.logger.info(f"   - AI 수익률: {return_rate:.2f}%")
    log_manager.logger.info(f"   - Buy-and-Hold 수익률: {buy_and_hold_return_rate:.2f}%")
    log_manager.logger.info(f"   - AI 초과 수익률: {ai_vs_buy_hold_excess:.2f}%p")
    
    if ai_vs_buy_hold_excess > 0:
        log_manager.logger.info(f"🎉 AI 모델이 Buy-and-Hold 전략보다 {ai_vs_buy_hold_excess:.2f}%p 더 좋은 성과를 보였습니다!")
    else:
        log_manager.logger.info(f"😞 AI 모델이 Buy-and-Hold 전략보다 {abs(ai_vs_buy_hold_excess):.2f}%p 낮은 성과를 보였습니다.")

    # ✅ 그래프 생성 및 표시
    log_manager.logger.info("📊 AI 포트폴리오와 Buy-and-Hold 전략 비교 그래프를 생성합니다...")
    
    # 그래프 설정 - 단일 y축 사용
    plt.figure(figsize=(15, 10))
    
    # AI 포트폴리오와 Buy-and-Hold 포트폴리오 비교
    plt.plot(tracking_dates, portfolio_values, 'b-', linewidth=2, label='AI Portfolio')
    plt.plot(tracking_dates, buy_and_hold_values, 'r-', linewidth=2, label='Buy-and-Hold Strategy')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (KRW)', fontsize=12)
    plt.title('AI Portfolio vs Buy-and-Hold Strategy Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # x축 날짜 형식 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    
    # 그래프 저장
    plt.tight_layout()
    chart_path = "output/ai_vs_buy_hold_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    log_manager.logger.info(f"📈 그래프가 저장되었습니다: {chart_path}")
    
    # 정규화된 비교 그래프도 생성
    plt.figure(figsize=(15, 8))
    
    # 정규화를 위해 초기값으로 값들을 나누어 비율로 표시
    portfolio_normalized = np.array(portfolio_values) / initial_portfolio_value
    buy_and_hold_normalized = np.array(buy_and_hold_values) / initial_portfolio_value
    
    plt.plot(tracking_dates, portfolio_normalized, 'b-', linewidth=2, label='AI Portfolio (Normalized)')
    plt.plot(tracking_dates, buy_and_hold_normalized, 'r-', linewidth=2, label='Buy-and-Hold Strategy (Normalized)')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Performance Ratio (Initial = 1.0)', fontsize=12)
    plt.title('AI Portfolio vs Buy-and-Hold Strategy Performance (Normalized)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # x축 날짜 형식 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    normalized_chart_path = "output/ai_vs_buy_hold_normalized.png"
    plt.savefig(normalized_chart_path, dpi=300, bbox_inches='tight')
    log_manager.logger.info(f"📈 정규화 그래프가 저장되었습니다: {normalized_chart_path}")
    
    # 그래프 표시 (옵션)
    plt.show()

    # ✅ 데이터프레임으로 변환 및 출력
    pd.set_option("display.max_rows", None)
    # ✅ 데이터프레임으로 변환 및 상·하위 5개만 출력
    result_df = pd.DataFrame(predictions, columns=["날짜", "예측 매매 결정", "확률(%)"])

    log_manager.logger.info("📌 예측 결과 (상위 5개)")
    log_manager.logger.info(result_df.head(5))

    log_manager.logger.info("📌 예측 결과 (하위 5개)")
    log_manager.logger.info(result_df.tail(5))

    # 각 매매 결정별 총 개수 계산
    total_sell = result_df["예측 매매 결정"].str.startswith("매도").sum()
    total_hold = result_df["예측 매매 결정"].str.startswith("관망").sum()
    total_buy  = result_df["예측 매매 결정"].str.startswith("매수").sum()

    summary = f"총 매도: {total_sell}건, 총 관망: {total_hold}건, 총 매수: {total_buy}건"
    log_manager.logger.info(summary)

    # ✅ 확률 분포에서 매수/매도/관망 각각의 총합 계산
    buy_prob_sum = np.sum(probs[0][1:2])
    sell_prob_sum = np.sum(probs[0][2:3])
    hold_prob = probs[0][0]

    total_sum = buy_prob_sum + sell_prob_sum + hold_prob

    buy_percent = (buy_prob_sum / total_sum) * 100
    sell_percent = (sell_prob_sum / total_sum) * 100
    hold_percent = (hold_prob / total_sum) * 100

    log_manager.logger.info(f"📊 전체 확률 분포 요약:")
    log_manager.logger.info(f"🟩 매수(Buy) 확률 총합: {buy_prob_sum:.2f} ({buy_percent:.2f}%)")
    log_manager.logger.info(f"🟥 매도(Sell) 확률 총합: {sell_prob_sum:.2f} ({sell_percent:.2f}%)")
    log_manager.logger.info(f"🟨 관망(Hold) 확률: {hold_prob:.2f} ({hold_percent:.2f}%)")


    # ✅ 실제 데이터의 마지막 날짜 (df 기준)
    true_last_date = df.iloc[-1]['Date']

    # ✅ 예측 구간 기준 마지막 날짜 및 액션 결과
    predicted_last_date, last_action_str, _ = predictions[-1]
    last_action_index = action  # 마지막 루프에서 나온 action 값
    last_action_prob = probs[0][last_action_index]

    log_manager.logger.info(f"📅 예측 가능한 구간의 마지막 날짜: {predicted_last_date}")
    log_manager.logger.info(f"📈 마지막 예측 액션: {last_action_str} (확률: {last_action_prob:.2f}%)")

    # ✅ 액션 유형별 상세 로그
    if last_action_str.startswith("매수"):
        shares_bought = int(last_action_str.split()[1].replace("주", ""))
        log_manager.logger.info(f"🛒 마지막 시점에서 {shares_bought}주 매수 예정 (확률: {last_action_prob:.2f}%)")
    elif last_action_str.startswith("매도"):
        shares_sold = int(last_action_str.split()[1].replace("주", ""))
        log_manager.logger.info(f"💰 마지막 시점에서 {shares_sold}주 매도 예정 (확률: {last_action_prob:.2f}%)")
    else:
        log_manager.logger.info(f"⏸ 마지막 시점에서는 관망(Hold) 상태입니다. (확률: {last_action_prob:.2f}%)")

    # 예시: 원하는 날짜 입력
    target_date = "2024-07-12"
    action_str, prob = get_prediction_by_date(result_df, target_date)

    if action_str is not None:
        log_manager.logger.info(f"📅 [{target_date}] 예측 결과: {action_str} (확률: {prob:.2f}%)")

        # ✅ 전체 액션 확률을 저장용 리스트로 정리
        # ✅ 전체 액션 확률을 저장용 리스트로 정리
    all_prob_records = []
    for i in range(len(predictions)):
        date = predictions[i][0]
        main_action = predictions[i][1]
        main_prob = predictions[i][2]
        prob_row = probs_list[i]  # ✅ 확률 분포

        row = {
            "날짜": date,
            "예측 매매 결정": main_action,
            "확률(%)": main_prob,
        }
        for j in range(len(prob_row)):
            row[f"Action_{j}_확률(%)"] = prob_row[j]
        all_prob_records.append(row)

    # ✅ 데이터프레임으로 변환
    detailed_result_df = pd.DataFrame(all_prob_records)

    # ✅ CSV로 저장
    output_csv_path = "output/prediction_result_detailed.csv"
    detailed_result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    log_manager.logger.info(f"📁 예측 결과가 CSV로 저장되었습니다: {output_csv_path}")



    # 예시 명령어
    # python main_predict.py --model_path output/ppo_stock_trader_episode_350.pth --test_data data/csv/005930.KS_combined_train_data.csv
    # python main_predict.py --model_path output/ppo_stock_trader_episode_1148.pth --test_data data/csv/005930.KS_combined_test_data.csv