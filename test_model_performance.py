import torch
import os
import pandas as pd
from models.transformer_model import StockTransformer
from data.data_loader import load_stock_data
from config import config_manager

def load_model(model_path: str, model_class, input_dim: int, device="cpu") -> torch.nn.Module:
    """
    저장된 모델 가중치를 불러오는 함수

    Args:
        model_path (str): 저장된 모델 경로
        model_class (torch.nn.Module): 모델 클래스 (StockTransformer 등)
        input_dim (int): 모델 입력 차원
        device (str, optional): 사용할 디바이스. 기본값: "cpu"

    Returns:
        torch.nn.Module: 불러온 모델 객체
    """
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return None

    try:
        model = model_class(input_dim=input_dim).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # strict=False로 일부 키 누락 방지
        model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None


def predict_action(model: torch.nn.Module, state, device="cpu"):
    """
    주어진 상태(state)에 대해 모델이 예측하는 매매 결정을 반환하는 함수

    Args:
        model (torch.nn.Module): 학습된 모델 객체
        state (np.array): 입력 상태 데이터 (최근 n개 시퀀스)
        device (str, optional): 실행할 디바이스. 기본값: "cpu"

    Returns:
        int: 예측된 액션 (0=매도, 1=관망, 2=매수)
    """
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = torch.softmax(model(state_tensor), dim=-1)
        action = torch.argmax(probs, dim=-1).item()
    return action


def execute_trading_strategy(stock_data, model, dates, prices, device="cpu"):
    """
    모델의 예측을 기반으로 매매 전략을 실행하는 함수

    Args:
        stock_data (np.array): 주가 데이터 (입력 피처)
        model (torch.nn.Module): 학습된 모델 객체
        dates (list): 날짜 리스트
        prices (list): 종가 리스트
        device (str, optional): 실행할 디바이스. 기본값: "cpu"

    Returns:
        pd.DataFrame: 매매 예측 결과 및 포트폴리오 가치
    """
    initial_balance = 10000000  # 초기 투자금 (USD)
    balance = initial_balance
    shares_held = 0
    transaction_fee = 0.001  # 거래 수수료 0.1%
    buy_and_hold_shares = initial_balance / (prices[0] * (1 + transaction_fee))
    buy_and_hold_value = buy_and_hold_shares * prices[-1]

    action_dict = {0: "매도(Sell)", 1: "관망(Hold)", 2: "매수(Buy)"}
    predictions = []
    portfolio_values = []

    observation_window = 30  # 과거 30개 데이터 기반 예측
    for i in range(len(stock_data)):
        if i < observation_window:  # 데이터 부족 시 스킵
            continue

        state = stock_data[i - observation_window : i]  # 최근 30개 데이터 사용
        date = dates[i]
        price = prices[i]

        action = predict_action(model, state, device)

        # ✅ 매매 로직 실행
        if action == 2 and balance > price * (1 + transaction_fee):  # 매수
            shares_to_buy = balance / (price * (1 + transaction_fee))
            shares_to_buy = int(shares_to_buy)
            cost = shares_to_buy * price * (1 + transaction_fee)
            balance -= cost
            shares_held += shares_to_buy
        elif action == 0 and shares_held > 0:  # 매도
            revenue = shares_held * price * (1 - transaction_fee)
            balance += revenue
            shares_held = 0

        # ✅ 포트폴리오 가치 업데이트
        portfolio_value = balance + (shares_held * price)
        portfolio_values.append(portfolio_value)
        predictions.append([date, action_dict[action], price, portfolio_value])

    # ✅ 최종 수익률 계산
    final_portfolio_value = balance + (shares_held * prices[-1])
    model_profit = ((final_portfolio_value - initial_balance) / initial_balance) * 100
    buy_and_hold_profit = ((buy_and_hold_value - initial_balance) / initial_balance) * 100

    print("\n📈 최종 포트폴리오 가치:", round(final_portfolio_value, 2), "USD")
    print("📊 Buy & Hold 포트폴리오 가치:", round(buy_and_hold_value, 2), "USD")
    print(f"🤖 모델 수익률: {round(model_profit, 2)}%")
    print(f"📉 Buy & Hold 수익률: {round(buy_and_hold_profit, 2)}%")

    return pd.DataFrame(predictions, columns=["날짜", "예측 매매 결정", "종가", "포트폴리오 가치"])


if __name__ == "__main__":
    """
    모델 평가 실행 부분
    """
    # ✅ 설정 불러오기
    device = torch.device(config_manager.get_device())

    # ✅ 모델 경로 설정
    MODEL_PATH = os.path.join("output", "ppo_stock_trader_episode_313.pth")  # 모델 파일명 변경 가능

    # ✅ 테스트 데이터 불러오기
    stock_data, input_dim = load_stock_data("data/csv/sp500_test_data.csv")
    df = pd.read_csv("data/csv/sp500_test_data.csv")
    dates = df["Date"].values
    prices = df["Close"].values

    # ✅ 모델 불러오기
    model = load_model(MODEL_PATH, StockTransformer, input_dim, device)

    if model is None:
        print("❌ 모델을 불러오지 못했습니다. 경로를 확인하세요.")
        exit(1)

    # ✅ 매매 전략 실행 및 결과 저장
    result_df = execute_trading_strategy(stock_data, model, dates, prices, device)

    # ✅ 결과 출력 및 저장
    print(result_df)
    result_df.to_csv("output/model_predictions.csv", index=False)
