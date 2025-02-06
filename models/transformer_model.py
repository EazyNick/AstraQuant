import torch
import torch.nn as nn

import os
import sys

current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..")) # 현재 디렉토리에 따라 이 부분 수정
sys.path.append(project_root)

from manage import PathManager
path_manager = PathManager()

# 원하는 경로 추가
sys.path.append(path_manager.get_path("config"))
sys.path.append(path_manager.get_path("logs"))

# import
try:
    from logs import log_manager
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")

class StockTransformer(nn.Module):
    def __init__(self, input_dim=None):
        super(StockTransformer, self).__init__()

        self.device = torch.device(config_manager.get_device())
        model_dim = config_manager.get_model_dim()
        num_heads = config_manager.get_num_heads()
        num_layers = config_manager.get_num_layers()

        # ✅ 입력 차원 설정 (기본값: config에서 불러오기)
        if input_dim is None:
            input_dim = config_manager.get_input_dim()

        # 입력 데이터를 Transformer 입력 차원으로 변환
        self.embedding = nn.Linear(input_dim, model_dim).to(self.device)

        # Transformer 인코더 레이어 설정 (batch_first=True 옵션 추가)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True).to(self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

        # 매수(Buy), 보유(Hold), 매도(Sell) 3가지 액션을 위한 최종 FC 레이어
        self.fc = nn.Linear(model_dim, 3).to(self.device)

    def forward(self, x):
        print(f"입력 데이터 초기 shape: {x.shape}")
        # x의 차원이 (batch, seq_len)이면, 마지막 차원을 추가해 (batch, seq_len, input_dim)으로 변환
        if x.dim() == 2:  # (seq_len, feature_dim) → batch 차원 없음
            x = x.unsqueeze(0)  # (1, seq_len, feature_dim)
            print(f"batch 차원 추가 후 shape: {x.shape}")

        x = x.to(self.device)  # 입력 데이터 GPU 이동
        x = self.embedding(x)  # (batch, seq_len, input_dim) -> (batch, seq_len, model_dim)
        x = self.transformer(x)  # Transformer 인코더를 통과

        output = self.fc(x[:, -1, :])  # 마지막 타임스텝의 출력을 사용하여 매매 신호 예측
        print(f"최종 출력 shape: {output.shape}")  # ✅ 최종 출력 shape 확인

        return output

# ✅ StockTransformer 단독 테스트
if __name__ == "__main__":
    model = StockTransformer()
    test_input = torch.randn(30, config_manager.get_input_dim())  # (batch, seq_len, feature_dim)
    output = model(test_input)  # Transformer 모델을 거친 후 결과 반환
    print("✅ 모델 출력 크기:", output.shape)  # ✅ (16, 3) → (batch_size, action_classes), 각 값은 Buy, Hold, Sell 확률을 의미
    # print("✅ output:", output)
