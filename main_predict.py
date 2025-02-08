import torch
import numpy as np
import os
from models.transformer_model import StockTransformer
from data.data_loader import load_stock_data
from config import config_manager  # ✅ 설정 불러오기

# ✅ 저장된 모델 불러오기
def load_model(model_path, input_dim, device="cpu"):
    model = StockTransformer(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # ✅ 평가 모드 설정
    return model

# ✅ 매매 결정 함수
def predict_action(model, state, device):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # ✅ (1, seq_len, feature_dim) 변환
    probs = torch.softmax(model(state), dim=-1)  # ✅ 확률 계산
    action = torch.argmax(probs, dim=-1).item()  # ✅ 가장 높은 확률의 액션 선택
    return action, probs.cpu().detach().numpy()  # ✅ 액션과 확률 반환

if __name__ == "__main__":
    # ✅ 설정 가져오기
    device = torch.device(config_manager.get_device())

    # ✅ 저장된 모델 로드
    model_path = os.path.join(os.path.dirname(__file__), "output", "ppo_stock_trader.pth")
    stock_data, input_dim = load_stock_data("data/csv/sp500_test_data.csv")  # ✅ 테스트 데이터 로드
    model = load_model(model_path, input_dim, device)

    # ✅ 마지막 observation_window 만큼의 데이터 가져오기
    observation_window = config_manager.get_observation_window()
    if stock_data.shape[0] < observation_window:
        raise ValueError(f"❌ 테스트 데이터가 너무 적습니다! (필요: {observation_window}, 제공됨: {stock_data.shape[0]})")

    last_state = stock_data[-observation_window:]  # ✅ 마지막 observation_window 가져오기

    # ✅ 모델을 사용하여 액션 예측
    action, probs = predict_action(model, last_state, device)

    # ✅ 액션 출력
    action_dict = {0: "매도(Sell)", 1: "관망(Hold)", 2: "매수(Buy)"}
    print(f"\n📌 예측된 매매 결정: {action_dict[action]} (확률: {probs[0]})")
