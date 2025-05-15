# 데이터 로더 모듈 - 주식 데이터 불러오기

import pandas as pd
import numpy as np

def load_stock_data(file_path: str) -> tuple[np.ndarray, int]:
    """
    CSV 파일에서 주식 데이터를 불러오는 함수

    Args:
        file_path (str): 불러올 CSV 파일 경로

    Returns:
        tuple: (numpy.ndarray, int) 변환된 데이터와 입력 피처 개수
    """
    # ✅ CSV 파일 로드
    df = pd.read_csv(file_path)

    # ✅ 'Close' 컬럼이 존재하면 /1000 해줌
    if 'Close' in df.columns:
        df['Close'] = df['Close'] / 1000.0
        print("✅ 'Close' 컬럼을 1/1000로 스케일링했습니다.")
    else:
        print("⚠️ 'Close' 컬럼이 없습니다. 스케일링 생략.")

    # ✅ Date 컬럼 제외
    df = df.drop('Date', axis=1)

    # ✅ 모든 컬럼을 float32로 변환
    for col in df.columns:
        df[col] = df[col].astype('float32')

    # ✅ Numpy 배열로 변환
    data = df.values

    # ✅ 입력 피처 개수 반환
    input_dim = data.shape[1]

    # ✅ 최종 피처 개수 출력
    print(f"📊 데이터 형태: {data.shape}")
    print(f"📐 입력 피처 개수: {input_dim}개")

    return data, input_dim

# ✅ 테스트 코드 추가
if __name__ == "__main__":
    import os

    # ✅ 샘플 CSV 파일 경로 설정
    sample_file = "data/cursor_csv/AMZN_train_data.csv"

    # ✅ 파일이 존재하는지 확인 후 로드
    if os.path.exists(sample_file):
        stock_data, input_dim = load_stock_data(sample_file)
        print(f"✅ 데이터 로드 완료! 데이터 Shape: {stock_data.shape}, 입력 피처 개수: {input_dim}")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {sample_file}")
