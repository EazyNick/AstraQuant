"""시퀀스의 순서를 네트워크가 인식할 수 있도록, 위치 정보를 벡터에 더해주는 역할"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 🔹 Positional Encoding 행렬 초기화
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 🔹 짝수 인덱스 (2i)에는 sin 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)

        # 🔹 홀수 인덱스 (2i+1)에는 cos 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 배치 차원 추가 (1, max_len, d_model)

        # 🔹 register_buffer: 학습되지 않는 상태값으로 등록 (GPU에서도 사용 가능)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # Positional Encoding 추가
        return self.dropout(x)
