import torch
import numpy as np
import os
import argparse
import pandas as pd
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
        if missing:
            print(f"⚠️ 누락된 가중치: {missing}")
        if unexpected:
            print(f"⚠️ 예상치 못한 키: {unexpected}")

        # for name, param in model.named_parameters():
        #     print(f"🔍 {name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")

        # ✅ 가중치 적용 (strict=False: 모델 구조가 일부 다를 경우 대비)
        # model.load_state_dict(state_dict, strict=False)

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
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq_len, feature_dim) 변환
    # with torch.no_grad():
    #     logits = model(state)
    #     print(f"🧐 모델 출력(logits) 샘플: {logits.cpu().numpy()[0][:10]}")  # 앞 10개 출력
    #     probs = torch.softmax(logits, dim=-1)
    #     print(f"📊 Softmax 확률 샘플: {probs.cpu().numpy()[0][:10]}")  # 앞 10개 확률
    probs = torch.softmax(model(state), dim=-1) # 확률 계산
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

    # ✅ 저장된 모델 로드 
    stock_data, input_dim = load_stock_data(args.test_data)
    actor_model = load_model(args.model_path, ActorNetwork, input_dim, device)
    # if model is None:
    #     raise ValueError("모델이 로드되지 않았습니다.")

    df = pd.read_csv(args.test_data or 'data/csv/sp500_test_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])  # 날짜 형식 변환
    df = df.sort_values('Date').reset_index(drop=True)
    dates = df['Date'].values  # 다시 정렬된 날짜로 업데이트

    log_manager.logger.info(f"📅 전체 테스트 데이터 마지막 날짜: {dates[-1]}")

    # ✅ 마지막 observation_window 만큼의 데이터 가져오기
    observation_window = config_manager.get_observation_window()
    if stock_data.shape[0] < observation_window:
        raise ValueError(f"❌ 테스트 데이터가 너무 적습니다! (필요: {observation_window}, 제공됨: {stock_data.shape[0]})")

    # ✅ 전체 데이터에 대한 예측 수행
    action_dict = {}
    max_volume = config_manager.get_max_shares_per_trade()
    # action_dict 생성
    action_dict[0] = "관망(Hold)"
    for i in range(1, max_volume + 1):
        action_dict[i] = f"매수(Buy) {i}주"
        action_dict[i + max_volume] = f"매도(Sell) {i}주"

    predictions = []
    # ✅ stock_data 크기만큼 앞에서 자르기
    if len(dates) > stock_data.shape[0]:
        dates = dates[-stock_data.shape[0]:]  # 뒤쪽 기준으로 자르기

    for i in range(observation_window, stock_data.shape[0]):
        state = stock_data[i - observation_window:i] # 관찰 윈도우 데이터 추출
        # ✅ 각 시점에 보유 수량을 붙임 (마지막 컬럼으로 추가)
        holding_column = np.full((observation_window, 1), holding, dtype=np.float32)
        state_with_holding = np.concatenate([state, holding_column], axis=1)
        date = dates[i] # 해당 날짜 가져오기
        action, probs = predict_action(actor_model, state_with_holding, device)
        predictions.append([date, action_dict[action], probs[0][action]])
        current_price = stock_data[i, 0] * 100

        # ✅ 보유 수량 업데이트
        if 1 <= action <= max_volume: # 매수
            cost = action * current_price * (1 + transaction_fee)
            if cost <= balance:
                holding += action
                balance -= cost
                print("매수: " + holding + "주")
        elif max_volume < action <= 2 * max_volume: # 매도
            sell_volume = min(holding, action - max_volume)
            revenue = sell_volume * current_price * (1 - transaction_fee)
            balance += revenue
            holding -= sell_volume
            print("매도: " + holding + "주")

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
    buy_prob_sum = np.sum(probs[0][1:max_volume + 1])
    sell_prob_sum = np.sum(probs[0][max_volume + 1:2 * max_volume + 1])
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
    target_date = "2020-03-27"
    action_str, prob = get_prediction_by_date(result_df, target_date)

    if action_str is not None:
        log_manager.logger.info(f"📅 [{target_date}] 예측 결과: {action_str} (확률: {prob:.2f}%)")
    # 예시 명령어
    # python main_predict.py --model_path output/ppo_stock_trader_episode_5.pth --test_data data/csv/005930.KS_combined_train_data.csv
