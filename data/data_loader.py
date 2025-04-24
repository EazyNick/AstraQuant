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
    # selected_columns = [col for col in df.columns if "Slope" in col or "Close" in col]

    # ✅ "Close" 포함하면서 W_Close, M_Close는 제외
    close_columns = [col for col in df.columns if "Close" in col and col not in ["W_Close", "M_Close"]]

    # ✅ Close 계열 값 스케일링 (1000으로 나누기)
    for col in close_columns:
        if col in df.columns:
            df[col] = df[col] / 100000.0

    # ✅ 남길 컬럼 리스트
    # selected_columns = close_columns + [
    #     "D_Slope_SMA_5", "D_Slope_SMA_10", "D_Slope_SMA_15", "D_Slope_SMA_20",
    #     "W_Slope_SMA_5", "W_Slope_SMA_10",
    #     "M_Slope_SMA_5"
    # ]

    selected_columns = close_columns + [
        "D_Slope_SMA_5", "D_Slope_SMA_10", "D_Slope_SMA_15", "D_Slope_SMA_20",
    ]

    # # ✅ 컬럼 이름 사전 정의
    # preferred_order = [col for col in df.columns if "Close" in col and col not in ["W_Close", "M_Close"]] + \
    #               [col for col in df.columns if "Slope" in col and "vma" not in col.lower()]

    # ✅ 교집합 유지하며 순서 보장
    # selected_columns = [col for col in preferred_order if col in df.columns]

    # ✅ 선택된 피처 출력
    print(f"📌 선택된 피처: {selected_columns}")

    # ✅ 먼저 선택된 컬럼만 추출
    df = df[selected_columns]

    # ✅ 각 Slope 컬럼에 대해 Z-score 정규화 또는 Tanh 변환 적용
    slope_columns = [col for col in df.columns if "Slope" in col]

    for col in slope_columns:
        if col in df.columns:
            # 방법 1: Z-score 정규화 (평균 0, 표준편차 1)
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0.0  # 표준편차가 0인 경우 정규화 불가 → 0 처리

            # 방법 2 (대안): Tanh 정규화
            # df[col] = np.tanh(df[col])

    # ✅ 0이 하나라도 있는 열 제거
    before_columns = df.columns.tolist()
    df = df.loc[(df != 0).all(axis=1)]  # ✅ 0이 있는 "행" 제거
    after_columns = df.columns.tolist()

    # ✅ 제거된 컬럼 확인
    removed_columns = list(set(before_columns) - set(after_columns))
    if removed_columns:
        print(f"🗑️ 제거된 컬럼 ({len(removed_columns)}개): {removed_columns}")
    else:
        print("✅ 모든 선택된 컬럼이 유지되었습니다.")

    # # ✅ Slope 값에만 Tanh 변환 적용
    # slope_columns = [col for col in df.columns if "Slope" in col]
    # # print(f"🎯 Tanh 변환 적용 열: {slope_columns}")  # 변환 대상 열 확인용 로그

    # ✅ 최종 선택된 열 출력
    print("📊 최종 변환된 데이터 열 및 샘플 데이터:")
    print(df.head())  # 데이터 일부 출력

    # ✅ Numpy 배열로 변환
    data = df.values

    # ✅ 입력 피처 개수 반환
    input_dim = data.shape[1]

    # ✅ 최종 피처 개수 출력
    print(f"📐 데이터에서 걸러진 피처 개수: {input_dim}개")

    return data, input_dim

# ✅ 테스트 코드 추가
if __name__ == "__main__":
    import os

    # ✅ 샘플 CSV 파일 경로 설정
    sample_file = "data/csv/005930.KS_combined_train_data.csv"

    # ✅ 파일이 존재하는지 확인 후 로드
    if os.path.exists(sample_file):
        stock_data, input_dim = load_stock_data(sample_file)
        print(f"✅ 데이터 로드 완료! 데이터 Shape: {stock_data.shape}, 입력 피처 개수: {input_dim}")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {sample_file}")

    # ✅ 정규화된 데이터를 CSV로 저장
    output_csv_path = "output/processed_stock_data.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # 폴더 없으면 생성
    pd.DataFrame(stock_data).to_csv(output_csv_path, index=False)
    print(f"📁 정규화된 데이터가 저장되었습니다: {output_csv_path}")
