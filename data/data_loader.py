# 데이터 로더 모듈 - 주식 데이터 불러오기

import pandas as pd

def load_stock_data(file_path):
    # CSV 파일 로드
    df = pd.read_csv(file_path)

    # 거래량(VMA_* 포함) 제외
    selected_columns = [col for col in df.columns if "Volume" not in col and "VMA" not in col]

    # 날짜 제외하고 numpy 배열로 변환
    data = df[selected_columns].drop(columns=['Date'], errors='ignore').values

    # 피처 개수 반환
    input_dim = data.shape[1]
    return data, input_dim
