# 데이터 로더 모듈 - 주식 데이터 불러오기

import pandas as pd
import numpy as np

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
    # True/False 변환 후, 숫자형 데이터만 float으로 변환
    df = df.replace({True: 1000.0, False: 0.0})
    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).astype(float)

    # 경고 메시지 안뜨게 함
    # # ✅ Boolean 값을 0과 1로 변환
    # df = df.astype(object).replace({True: 1.0, False: 0.0})  # 명시적으로 object 변환 후 처리
    # df = df.infer_objects(copy=False)  # 자동 형 변환 방식 지정 (FutureWarning 방지)

    # # ✅ 숫자형 데이터 변환 (기존 방식 유지)
    # df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number])

    # ✅ 이동평균선 제외, 기울기(Slope) 및 가격(Close)만 포함
    selected_columns = [col for col in df.columns if "Slope" in col or "Close" in col]

    # ✅ 선택된 칼럼명을 저장
    selected_feature_names = df[selected_columns].columns.tolist()
    print(f"📌 선택된 피처: {selected_feature_names}")

    # ✅ 날짜(Date) 컬럼 제외하고 데이터 변환
    df = df[selected_columns]


    # # ✅ Slope 값에만 Tanh 변환 적용
    # slope_columns = [col for col in df.columns if "Slope" in col]
    # # print(f"🎯 Tanh 변환 적용 열: {slope_columns}")  # 변환 대상 열 확인용 로그

    # # `tanh` 변환 적용 (Slope 값만)
    # df[slope_columns] = np.tanh(df[slope_columns])

    # ✅ Numpy 배열로 변환
    data = df.values

    # ✅ 입력 피처 개수 반환
    input_dim = data.shape[1]

    return data, input_dim

# ✅ 테스트 코드 추가
if __name__ == "__main__":
    import os

    # ✅ 샘플 CSV 파일 경로 설정
    sample_file = "data/csv/GSPC_combined_test_data.csv"

    # ✅ 파일이 존재하는지 확인 후 로드
    if os.path.exists(sample_file):
        stock_data, input_dim = load_stock_data(sample_file)
        print(f"✅ 데이터 로드 완료! 데이터 Shape: {stock_data.shape}, 입력 피처 개수: {input_dim}")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {sample_file}")
