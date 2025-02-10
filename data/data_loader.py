# 데이터 로더 모듈 - 주식 데이터 불러오기

import pandas as pd

def load_stock_data(file_path):
    """
    CSV 파일에서 주식 데이터를 불러오는 함수

    Args:
        file_path (str): 불러올 CSV 파일 경로

    Returns:
        tuple: (numpy.ndarray, int) 변환된 데이터와 입력 피처 개수
    """
    # ✅ CSV 파일 로드
    df = pd.read_csv(file_path)

    # ✅ NaN 값 처리
    df.fillna(0, inplace=True)

    # ✅ Boolean 값을 0과 1로 변환
    df = df.replace({True: 1.0, False: 0.0})

    # ✅ 거래량(VMA_* 포함) 컬럼 제거
    selected_columns = [col for col in df.columns if "Volume" not in col and "VMA" not in col]

    # ✅ 날짜(Date) 컬럼 제외하고 numpy 배열로 변환
    data = df[selected_columns].drop(columns=['Date'], errors='ignore').values

    # ✅ 입력 피처 개수 반환
    input_dim = data.shape[1]
    return data, input_dim

# ✅ 테스트 코드 추가
if __name__ == "__main__":
    import os

    # ✅ 샘플 CSV 파일 경로 설정
    sample_file = "data/csv/sp500_training_data.csv"

    # ✅ 파일이 존재하는지 확인 후 로드
    if os.path.exists(sample_file):
        stock_data, input_dim = load_stock_data(sample_file)
        print(f"✅ 데이터 로드 완료! 데이터 Shape: {stock_data.shape}, 입력 피처 개수: {input_dim}")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {sample_file}")
