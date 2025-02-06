import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

# 모델과 모든 텐서를 device로 변환하는 .to(self.device)를 모든 연산에서 명시적으로 적용해야 함

class PPOAgent:
    def __init__(self, model):
        self.device = torch.device(config_manager.get_device())  # ✅ device 설정

        self.model = model.to(self.device)  # ✅ 모델을 GPU/CPU로 이동
        self.optimizer = optim.Adam(self.model.parameters(), lr=config_manager.get_learning_rate())
        self.gamma = config_manager.get_gamma()
        self.epsilon = config_manager.get_epsilon()
        self.batch_size = config_manager.get_batch_size()
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, feature_dim)
        probs = torch.softmax(self.model(state), dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, memory):
        states, actions, rewards = zip(*memory)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        discounted_rewards = []
        sum_reward = 0
        for r in reversed(rewards):
            sum_reward = r + self.gamma * sum_reward
            discounted_rewards.insert(0, sum_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)

        for i in range(0, len(states), self.batch_size):
            batch_states = states[i:i+self.batch_size]
            batch_actions = actions[i:i+self.batch_size]
            batch_rewards = discounted_rewards[i:i+self.batch_size]

            probs = torch.softmax(self.model(batch_states), dim=-1)
            action_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze()
            old_probs = action_probs.detach()

            ratio = action_probs / old_probs
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            loss = -torch.min(ratio * batch_rewards, clipped_ratio * batch_rewards).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    from models.transformer_model import StockTransformer  # 모델 임포트
    # ✅ 설정 가져오기
    device = torch.device(config_manager.get_device())
    input_dim = config_manager.get_input_dim()
    seq_len = 30  # 시계열 길이

    # ✅ 가짜 데이터(Mock Data) 생성 (랜덤 데이터)
    batch_size = config_manager.get_batch_size()
    test_states = torch.randn(batch_size, seq_len, input_dim).to(device)  # (batch, seq_len, feature_dim)
    
    # ✅ Transformer 모델 생성 및 PPOAgent 초기화
    model = StockTransformer(input_dim=input_dim).to(device)
    agent = PPOAgent(model)

    # ✅ 액션 선택 테스트
    print("\n🎯 액션 선택 테스트:")
    test_state = test_states[0].cpu().numpy()  # 단일 샘플 (CPU로 변환하여 테스트)
    action = agent.select_action(test_state)
    print(f"🔹 선택된 액션: {action}")  # 0 (매도), 1 (보유), 2 (매수)

    # ✅ 업데이트 테스트 (가짜 메모리 데이터)
    print("\n📌 에이전트 업데이트 테스트:")
    test_memory = [
        (test_states[i].cpu().numpy(), np.random.randint(0, 3), np.random.randn()) 
        for i in range(batch_size)
    ]  # (state, action, reward) 랜덤 데이터 생성

    print(f"📝 메모리 샘플 개수: {len(test_memory)}")
    agent.update(test_memory)
    print("✅ 에이전트 업데이트 완료!")

