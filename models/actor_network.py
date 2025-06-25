import torch
import torch.nn as nn
from models.positionalencoding import PositionalEncoding
from config import config_manager

class ActorNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ActorNetwork, self).__init__()
        self.device = torch.device(config_manager.get_device())

        model_dim = config_manager.get_model_dim()
        num_heads = config_manager.get_num_heads()
        num_layers = config_manager.get_num_layers()

        # 🔥 액션 차원을 3개로 고정: 0=관망, 1=전부매수, 2=전부매도
        self.action_dim = 3
        self.input_dim = input_dim  # main_train.py에서 이미 input_dim + 1을 전달함

        self.embedding = nn.Linear(self.input_dim, model_dim).to(self.device)
        self.positional_encoding = PositionalEncoding(model_dim).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True).to(self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

        self.fc = nn.Linear(model_dim, self.action_dim).to(self.device)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        # 🔥 입력 데이터 확인 로그 (1000 스텝마다)
        if hasattr(self, 'train_step'):
            if self.train_step % 1000 == 0:
                from logs import log_manager
                log_manager.logger.debug(
                    f"[Actor] 입력 데이터 확인:\n"
                    f"  - 입력 shape: {x.shape}\n"
                    f"  - Feature 개수: {x.shape[-1]}\n"
                    f"  - 마지막 feature (보유 주식수): {x[0, :, -1]}\n"
                    f"  - 보유 주식수 평균: {x[0, :, -1].mean().item():.4f}"
                )
        
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = torch.clamp(x, -10, 10)
        logits = self.fc(x[:, -1, :])
        return logits
