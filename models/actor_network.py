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

        self.max_shares_per_trade = config_manager.get_max_shares_per_trade()
        self.action_dim = 1 + 2 * self.max_shares_per_trade
        self.input_dim = input_dim + 1  # 보유 주식 수 포함

        self.embedding = nn.Linear(self.input_dim, model_dim).to(self.device)
        self.positional_encoding = PositionalEncoding(model_dim).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True).to(self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(self.device)

        self.fc = nn.Linear(model_dim, self.action_dim).to(self.device)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = torch.clamp(x, -10, 10)
        logits = self.fc(x[:, -1, :])
        return logits
