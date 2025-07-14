"""
모델 로딩 관련 함수들
"""

import torch
import os


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

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"📐 모델 입력 차원: {input_dim}")
        if missing:
            print(f"⚠️ 누락된 가중치: {missing}")
        if unexpected:
            print(f"⚠️ 예상치 못한 키: {unexpected}")

        # ✅ 모델 평가 모드 설정
        model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")

        return model
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None 