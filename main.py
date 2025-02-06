# 프로그램 진입점 - 주요 모듈 연결 및 학습 실행

# conda activate AstraQuant

from env.stock_env import StockTradingEnv
from models.transformer_model import StockTransformer
from agents.ppo_agent import PPOAgent
from training.train import train_agent
from data.data_loader import load_stock_data
from config import config_manager  # YAML 설정 불러오기
import torch

# 설정값 불러오기
config = load_config()

# 학습 장치 설정 (CUDA 또는 CPU)
device = torch.device(config["general"]["device"])
print(f"✅ 학습 장치 설정: {device}")

# 데이터 로드 (input_dim 자동 추출)
stock_prices, input_dim = load_stock_data("data/csv/sp500_training_data.csv")

# 환경 및 모델 생성
env = StockTradingEnv(stock_prices, initial_balance=config["env"]["initial_balance"])
model = StockTransformer(input_dim=input_dim).to(device)  # 모델을 GPU/CPU로 이동
agent = PPOAgent(model)

# 학습 시작
train_agent(env, agent, episodes=config["training"]["episodes"])
