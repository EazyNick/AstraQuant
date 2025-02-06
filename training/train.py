from env.stock_env import StockTradingEnv
from models.transformer_model import StockTransformer
from agents.ppo_agent import PPOAgent
from data.data_loader import load_stock_data
from config import config_manager
import torch
import numpy as np

def train_agent(env, agent, episodes):
    device = torch.device(config_manager.get_device())  # ✅ GPU/CPU 설정

    for episode in range(episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        memory = []
        total_reward = 0

        for t in range(len(env.stock_data) - 30):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

            memory.append((state.cpu().numpy(), action, reward))  # PPO 학습을 위해 CPU로 변환
            state = next_state
            total_reward += reward

            if done or len(memory) >= agent.batch_size:
                agent.update(memory)
                memory = []  # 배치 학습 후 메모리 초기화

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    # ✅ config_manager는 어디서든 import 후 바로 사용 가능
    stock_prices, input_dim = load_stock_data("data/csv/sp500_training_data.csv")

    env = StockTradingEnv(stock_prices, initial_balance=config_manager.get_initial_balance())
    model = StockTransformer(input_dim=input_dim)
    agent = PPOAgent(model)

    train_agent(env, agent, episodes=config_manager.get_episodes())
