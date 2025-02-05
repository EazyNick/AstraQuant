import torch
import torch.nn as nn

class StockTransformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=2, num_layers=2):
        super(StockTransformer, self).__init__()

        # 입력 데이터를 Transformer 입력 차원으로 변환
        self.embedding = nn.Linear(input_dim, model_dim)

        # Transformer 인코더 레이어 설정 (batch_first=True 옵션 추가)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 매수(Buy), 보유(Hold), 매도(Sell) 3가지 액션을 위한 최종 FC 레이어
        self.fc = nn.Linear(model_dim, 3)

    def forward(self, x):
        # x의 차원이 (batch, seq_len)이면, 마지막 차원을 추가해 (batch, seq_len, input_dim)으로 변환
        if x.dim() == 2:  # (batch_size, seq_len)
            x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)

        x = self.embedding(x)  # (batch, seq_len, input_dim) -> (batch, seq_len, model_dim)
        x = self.transformer(x)  # Transformer 인코더를 통과
        return self.fc(x[:, -1, :])  # 마지막 타임스텝의 출력을 사용하여 매매 신호 예측
