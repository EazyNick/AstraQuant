import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent:
    def __init__(self, model, lr=0.0003, gamma=0.99, epsilon=0.2):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = torch.softmax(self.model(state), dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, memory):
        states, actions, rewards = zip(*memory)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Discounted reward 계산
        discounted_rewards = []
        sum_reward = 0
        for r in reversed(rewards):
            sum_reward = r + self.gamma * sum_reward
            discounted_rewards.insert(0, sum_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # PPO Loss 계산
        probs = torch.softmax(self.model(states), dim=-1)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
        old_probs = action_probs.detach()

        ratio = action_probs / old_probs
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * discounted_rewards, clipped_ratio * discounted_rewards).mean()

        # 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
