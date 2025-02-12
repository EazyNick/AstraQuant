import torch
import numpy as np
import pandas as pd
import os
from models.transformer_model import StockTransformer
from data.data_loader import load_stock_data
from config import config_manager

try:
    from logs import log_manager
except Exception as e:
    print(f"임포트 실패: {e}")

# ✅ 모델 저장 후, 다시 불러와서 확인
model = StockTransformer(input_dim=1)
torch.save(model, "output/test_model.pth", weights_only=False)

# ✅ 모델 불러오기 테스트
loaded_model = torch.load("output/test_model.pth")
print(loaded_model)  # 모델 구조 확인

