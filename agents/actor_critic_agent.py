"""
(1) 상태 입력 (state) ─▶
(2) actor(state) → action, log_prob
(3) critic(state) → value
(4) 보상 reward + (log_prob, value) 저장
(5) 여러 transition을 모아서 → 학습

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Categorical
from config import config_manager
from logs import log_manager

class ActorCriticAgent:
    """Actor-Critic 방식의 PPO 에이전트"""

    def __init__(self, actor, critic, writer=None):
        """
        Actor-Critic Agent 초기화

        Args:
            actor (nn.Module): 정책 네트워크 (Actor)
            critic (nn.Module): 가치 네트워크 (Critic)
            writer (SummaryWriter, optional): TensorBoard 기록용 writer
        """
        self.device = torch.device(config_manager.get_device())
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=config_manager.get_learning_rate())
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config_manager.get_learning_rate())

        self.gamma = config_manager.get_gamma()
        self.clampepsilon = config_manager.get_clampepsilon()
        self.batch_size = config_manager.get_batch_size()
        self.entropy_coef = 0.0 # 엔트로피 보상 계수, 값을 키울수록 정책이 평평해짐. 0.02 등 값 권장
        self.temperature = 1.0 # 정책 분포의 날카로움/평평함 조정, 1.0보다 클 경우 더욱 평평해짐(탐험 증가)

        self.writer = writer
        self.train_step = 0
        self.epsilon = config_manager.get_epsilon()
        self.epsilon_min = config_manager.get_epsilon_min()
        self.epsilon_decay = config_manager.get_epsilon_decay()
        self.max_shares_per_trade = config_manager.get_max_shares_per_trade()
        self.action_dim = actor.action_dim

    def select_action(self, state):
        """
        현재 상태에서 액션을 선택하고 로그 확률 및 상태 가치 반환

        Args:
            state (np.ndarray): 환경으로부터 받은 현재 상태 (shape: [seq_len, feature_dim])

        Returns:
            tuple: (action (int), log_prob (float), value (float))
        """
        """현재 상태에서 액터를 통해 행동을 선택하고 크리틱의 상태 가치를 함께 반환"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        if not torch.isfinite(state_tensor).all():
            print("⚠️ Invalid state detected:", state_tensor)

        logits = self.actor(state_tensor)
        probs = torch.softmax(logits / self.temperature, dim=-1)
        dist = Categorical(probs)

        if random.random() < self.epsilon:
            action = random.choice(range(self.action_dim))
        else:
            action = dist.sample().item()

        log_prob = dist.log_prob(torch.tensor(action).to(self.device))
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # ✅ TensorBoard 기록
        if self.writer and self.action_dim <= 10:
            for i in range(self.action_dim):
                self.writer.add_scalar(f"Action_Prob/Action_{i}", probs[0, i].item(), self.train_step)

        # ✅ 디버깅 로그
        if self.train_step % 100 == 0:
            topk = sorted(enumerate(probs[0].tolist()), key=lambda x: x[1], reverse=True)[:3]
            topk_log = {}
            for idx, val in topk:
                if idx == 0:
                    action_type = "관망"
                    action_info = f"{action_type}"
                elif 1 <= idx <= self.max_shares_per_trade:
                    action_type = "매수"
                    action_info = f"{action_type}({idx}주)"
                else:
                    action_type = "매도"
                    action_info = f"{action_type}({idx - self.max_shares_per_trade}주)"
                topk_log[f"Action_{idx}"] = f"{val:.4f} → {action_info}"
            log_manager.logger.debug(f"{self.train_step} step Top-3 probs:\n{topk_log}")

        self.train_step += 1
        value = self.critic(state_tensor).item()
        return action, log_prob.item(), value

    def update(self, memory):
        """
        저장된 에피소드 메모리를 기반으로 정책 및 가치 함수 업데이트

        Args:
            memory (list): (state, action, reward, log_prob, value) 튜플의 리스트
        """
        """메모리로부터 Advantage 기반 Actor-Critic 업데이트 수행"""
        states, actions, rewards, log_probs, values = zip(*memory)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        old_log_probs = torch.tensor(log_probs).to(self.device)
        values = torch.tensor(values).to(self.device)

        discounted_rewards = []
        running_add = 0
        for r in reversed(rewards):
            running_add = r + self.gamma * running_add
            discounted_rewards.insert(0, running_add)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)

        advantages = discounted_rewards - values # critic이 예측한 values와 실제 return의 차이

        for i in range(0, len(states), self.batch_size):
            b_states = states[i:i+self.batch_size]
            b_actions = actions[i:i+self.batch_size]
            b_advantages = advantages[i:i+self.batch_size].detach()
            b_returns = discounted_rewards[i:i+self.batch_size].detach() # 실제 보상 누적 값
            b_old_log_probs = old_log_probs[i:i+self.batch_size].detach()

            logits = self.actor(b_states)
            probs = torch.softmax(logits / self.temperature, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(b_actions)

            ratio = torch.exp(new_log_probs - b_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clampepsilon, 1 + self.clampepsilon)

            actor_loss = -torch.min(ratio * b_advantages, clipped_ratio * b_advantages).mean() # actor는 advantage 값을 기준으로 확률을 조정
            actor_loss -= self.entropy_coef * dist.entropy().mean()

            values_pred = self.critic(b_states).squeeze() # 현재 상태에서 critic이 예측한 값
            critic_loss = nn.MSELoss()(values_pred, b_returns) # 상태의 실제 보상 누적값과 자신이 예측한  V(s)의 차이를 줄이도록 학습

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            # ✅ TensorBoard 기록
            if self.writer:
                self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.train_step)
                self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.train_step)
                self.writer.add_scalar("Advantage/Mean", b_advantages.mean().item(), self.train_step)
                self.writer.add_scalar("Returns/Mean", b_returns.mean().item(), self.train_step)

            self.train_step += 1
