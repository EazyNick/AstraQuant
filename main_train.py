# 프로그램 진입점 - 주요 모듈 연결 및 학습 실행

# conda activate AstraQuant

# tensorboard --logdir=logs/trading 파이썬 3.13버전에서는 텐서보드 안됨
# tensorboard --logdir=logs

import os, sys, shutil
from env.stock_env import StockTradingEnv
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from agents.actor_critic_agent import ActorCriticAgent
from training.train import TrainingManager, train_agent
from data.data_loader import load_stock_data
from config import config_manager  # 싱글턴 ConfigManager 사용
import torch
from torch.utils.tensorboard import SummaryWriter

current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..")) # 현재 디렉토리에 따라 이 부분 수정
sys.path.append(project_root)

from manage import PathManager
path_manager = PathManager()

# 원하는 경로 추가
sys.path.append(path_manager.get_path("config"))
sys.path.append(path_manager.get_path("logs"))

try:
    from logs import log_manager
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")

# ✅ 설정값 불러오기
device = torch.device(config_manager.get_device())
log_manager.logger.info(f"✅ 학습 장치 설정: {device}")

# ✅ 데이터 로드 (input_dim 자동 추출)
log_manager.logger.info("데이터 불러오기...")
# stock_prices, input_dim = load_stock_data("data/csv/005930.KS_combined_train_data.csv")
stock_prices, input_dim = load_stock_data("data/cursor_csv/AMZN_train_data.csv")

# ✅ 기존 텐서보드 로그 삭제
log_dir = "logs/training"
if os.path.exists(log_dir):
    log_manager.logger.info(f"📁 기존 텐서보드 로그 디렉토리 삭제: {log_dir}")
    shutil.rmtree(log_dir)

# ✅ 텐서보드 로그 기록 시작
writer = SummaryWriter(log_dir=log_dir)

# ✅ 환경 및 모델 생성 (config.yaml에서 설정값 자동 적용)
env = StockTradingEnv(stock_prices, writer=writer)
actor = ActorNetwork(input_dim=input_dim)  # 보유 주식수 feature 포함
critic = CriticNetwork(input_dim=input_dim)  # 보유 주식수 feature 포함
agent = ActorCriticAgent(actor, critic, writer=writer)
# ✅ 정확한 입력 피처 개수 로그 출력 (보유 주식 수 포함된 input_dim)
log_manager.logger.info(f"📐 모델 입력 피처 개수 (보유 수량 포함): {input_dim + 1}")

training_manager = TrainingManager()
# ✅ 학습 시작
train_agent(env, agent, episodes=config_manager.get_episodes(), training_manager=training_manager)
