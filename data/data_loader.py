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

    close_price_scale = 10.0 # 학습을 위해 현재가 스케일링(나눠줌)

    # ✅ 'Close' 컬럼이 존재하면 /1000 해줌
    if 'Close' in df.columns:
        df['Close'] = df['Close'] / close_price_scale
        print("✅ 'Close' 컬럼을 1/10로 스케일링했습니다.")
    else:
        print("⚠️ 'Close' 컬럼이 없습니다. 스케일링 생략.")

    # ✅ Date 컬럼 제외
    df = df.drop('Date', axis=1)

    # ✅ NaN 값 체크 및 제거
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"⚠️ 데이터에 {nan_count}개의 NaN 값이 발견되었습니다. 제거합니다.")
        df = df.dropna()
        print(f"✅ NaN 제거 후 데이터 형태: {df.shape}")

    # ✅ 무한대 값 체크 및 제거
    inf_count = np.isinf(df.values).sum()
    if inf_count > 0:
        print(f"⚠️ 데이터에 {inf_count}개의 무한대 값이 발견되었습니다. 제거합니다.")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        print(f"✅ 무한대 값 제거 후 데이터 형태: {df.shape}")

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
        print(f"📂 CSV 파일 내용 미리보기: {sample_file}")
        df_preview = pd.read_csv(sample_file)
        print(df_preview.head(10))  # 처음 10줄 출력

        print("🔍 4번째 열 미리보기:")
        print(df_preview.iloc[:, 0].head(10))  # 4번째 열만 출력 (0-based index)

        stock_data, input_dim = load_stock_data(sample_file)
        print(f"✅ 데이터 로드 완료! 데이터 Shape: {stock_data.shape}, 입력 피처 개수: {input_dim}")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {sample_file}")
