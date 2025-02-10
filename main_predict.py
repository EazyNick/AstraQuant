import torch
import numpy as np
import os
from models.transformer_model import StockTransformer
from data.data_loader import load_stock_data
from config import config_manager 

try:
    from logs import log_manager
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")

# 저장된 모델 불러오기
def load_model(model_path, input_dim, device="cpu"):
    model = StockTransformer(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드 설정
    return model

# 확률 값 변환 함수 (0~100% 범위로 변환 및 최소값 보장)
def format_probs(probs):
    normalized_probs = probs * 100  # 확률을 0~100 범위로 변환
    formatted_probs = np.maximum(normalized_probs, 0.01)  # 최소값 0.01% 보장
    return np.round(formatted_probs, 2)  # 소수점 2자리까지 변환하여 출력

# 매매 결정 함수
def predict_action(model, state, device):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, feature_dim) 변환
    probs = torch.softmax(model(state), dim=-1)  # 확률 계산
    action = torch.argmax(probs, dim=-1).item()  # 가장 높은 확률의 액션 선택
    return action, format_probs(probs.cpu().detach().numpy())  # 액션과 확률 반환

if __name__ == "__main__":
    import pandas as pd
    # ✅ 설정 가져오기
    device = torch.device(config_manager.get_device())

    # ✅ 저장된 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), "output", "ppo_stock_trader_episode_230.pth")
    stock_data, input_dim = load_stock_data("data/csv/sp500_test_data.csv")  # ✅ 테스트 데이터 로드
    model = load_model(model_path, input_dim, device)

    # ✅ 날짜 및 피처 데이터 분리
    df = pd.read_csv("data/csv/sp500_test_data.csv")
    dates = df["Date"].values  # ✅ 날짜 데이터 저장

    # ✅ 마지막 observation_window 만큼의 데이터 가져오기
    observation_window = config_manager.get_observation_window()
    if stock_data.shape[0] < observation_window:
        raise ValueError(f"❌ 테스트 데이터가 너무 적습니다! (필요: {observation_window}, 제공됨: {stock_data.shape[0]})")

    # ✅ 전체 데이터에 대한 예측 수행
    action_dict = {0: "매도(Sell)", 1: "관망(Hold)", 2: "매수(Buy)"}
    predictions = []

    for i in range(observation_window, stock_data.shape[0]):
        state = stock_data[i - observation_window:i]  # 관찰 윈도우 데이터 추출
        date = dates[i]  # 해당 날짜 가져오기
        action, probs = predict_action(model, state, device)
        predictions.append([date, action_dict[action], probs[0]])

    # ✅ 데이터프레임으로 변환 및 출력
    result_df = pd.DataFrame(predictions, columns=["날짜", "예측 매매 결정", "확률(%)"])
    print(result_df)
