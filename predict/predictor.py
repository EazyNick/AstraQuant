"""
예측 관련 함수들
"""

import torch
import numpy as np
import pandas as pd


def format_probs(probs):
    """
    확률 값 변환 함수 (0~100% 범위로 변환 및 최소값 보장)
    """
    normalized_probs = probs * 100  # 확률을 0~100 범위로 변환
    formatted_probs = np.maximum(normalized_probs, 0.01)  # 최소값 0.01% 보장
    return np.round(formatted_probs, 2)  # 소수점 2자리까지 변환하여 출력


def predict_action(model, state, device):
    """
    매매 결정 함수
    """
    if model is None:
        raise ValueError("모델이 로드되지 않았습니다. 모델을 먼저 로드해주세요.")
    
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, feature_dim) 변환
    
    with torch.no_grad():
        logits = model(state)
        probs = torch.softmax(logits, dim=-1)  # 확률 계산
        action = torch.argmax(probs, dim=-1).item()  # 가장 높은 확률의 액션 선택
        return action, format_probs(probs.cpu().detach().numpy())  # 액션과 확률 반환


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