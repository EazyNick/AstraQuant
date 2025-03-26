import torch
import numpy as np
import os
import argparse
from models.transformer_model import StockTransformer
from data.data_loader import load_stock_data
from config import config_manager 

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

        # ✅ 가중치 적용 (strict=False: 모델 구조가 일부 다를 경우 대비)
        model.load_state_dict(state_dict, strict=False)

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
    probs = torch.softmax(model(state), dim=-1) # 확률 계산
    action = torch.argmax(probs, dim=-1).item() # 가장 높은 확률의 액션 선택
    return action, format_probs(probs.cpu().detach().numpy()) # 액션과 확률 반환

if __name__ == "__main__":
    import pandas as pd
    # ✅ 설정 가져오기
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(__file__), 'output', 'ppo_stock_trader_episode_296.pth'), help='사용할 모델 가중치 파일 (.pth) 경로 (기본값: ppo_stock_trader_episode_288.pth)')
    parser.add_argument('--test_data', type=str, default='data/csv/sp500_test_data.csv', help='테스트 데이터 (.csv) 파일 경로 (기본값: data/csv/sp500_test_data.csv)')
    args = parser.parse_args()

    device = torch.device(config_manager.get_device())
    # ✅ 초기 보유 수량
    holding = 0

    # ✅ 저장된 모델 로드 
    stock_data, input_dim = load_stock_data(args.test_data)
    model = load_model(args.model_path, StockTransformer, input_dim, device)
    # if model is None:
    #     raise ValueError("모델이 로드되지 않았습니다.")

    df = pd.read_csv(args.test_data or 'data/csv/sp500_test_data.csv')
    dates = df['Date'].values # ✅ 날짜 데이터 저장

    # ✅ 마지막 observation_window 만큼의 데이터 가져오기
    observation_window = config_manager.get_observation_window()
    if stock_data.shape[0] < observation_window:
        raise ValueError(f"❌ 테스트 데이터가 너무 적습니다! (필요: {observation_window}, 제공됨: {stock_data.shape[0]})")

    # ✅ 전체 데이터에 대한 예측 수행
    action_dict = {0: "매도(Sell)", 1: "관망(Hold)", 2: "매수(Buy)"}
    predictions = []

    for i in range(observation_window, stock_data.shape[0]):
        state = stock_data[i - observation_window:i] # 관찰 윈도우 데이터 추출
        # ✅ 각 시점에 보유 수량을 붙임 (마지막 컬럼으로 추가)
        holding_column = np.full((observation_window, 1), holding, dtype=np.float32)
        state_with_holding = np.concatenate([state, holding_column], axis=1)
        date = dates[i] # 해당 날짜 가져오기
        action, probs = predict_action(model, state_with_holding, device)
        predictions.append([date, action_dict[action], probs[0]])

        # ✅ 보유 수량 업데이트
        if action == 2:  # 매수
            holding += 8000
        elif action == 0 and holding > 0:  # 매도
            holding -= 8000

    # ✅ 데이터프레임으로 변환 및 출력
    # ✅ 모든 행을 출력하도록 설정 변경
    pd.set_option("display.max_rows", None)
    result_df = pd.DataFrame(predictions, columns=["날짜", "예측 매매 결정", "확률(%)"])
    log_manager.logger.info(result_df)

    # ✅ 각 매매 결정별 총 개수 계산 및 출력
    action_counts = result_df["예측 매매 결정"].value_counts()
    total_sell = action_counts.get("매도(Sell)", 0)
    total_hold = action_counts.get("관망(Hold)", 0)
    total_buy  = action_counts.get("매수(Buy)", 0)

    summary = f"총 매도: {total_sell}건, 총 관망: {total_hold}건, 총 매수: {total_buy}건"
    log_manager.logger.info(summary)

    # 예시 명령어
    # python main_predict.py --model_path output/ppo_stock_trader_episode_147.pth --test_data data/csv/GSPC_combined_test_data.csv