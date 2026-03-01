"""
[Workflow 흐름 요약]
(1) 환경에서 상태(state: [seq_len, feature_dim])를 받는다
(2) select_action(state):
    - state를 Tensor로 변환해 device로 보낸다
    - actor(state)로 logits -> softmax(temperature 적용) 확률 분포(probs) 생성
    - epsilon-greedy로 (랜덤 탐험 or dist.sample 기반 정책 샘플링) action 선택
    - 선택한 action의 log_prob를 계산하고, critic(state)로 value(V(s))를 추정한다
    - (action, log_prob, value)를 반환한다
(3) 외부에서 reward를 받고 transition (state, action, reward, log_prob, value)를 memory에 저장한다
(4) update(memory):
    - memory를 텐서로 변환
    - discounted_returns(누적 보상) 계산
    - advantage = returns - values 계산 후 정규화
    - mini-batch로 PPO actor 손실(클리핑 + 엔트로피) 및 critic 손실(MSE)로 학습한다
    - TensorBoard에 loss/advantage/returns 등을 기록한다

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


class ActorCriticAgent:  # Actor-Critic PPO 에이전트 클래스 정의
    """Actor-Critic 방식의 PPO 에이전트"""  # (기존 주석 유지) 클래스 설명

    def __init__(
        self, actor, critic, writer=None
    ):  # 생성자: actor/critic 네트워크 및 writer 입력
        """
        Actor-Critic Agent 초기화  # (기존 주석 유지)

        Args:
            actor (nn.Module): 정책 네트워크 (Actor)  # (기존 주석 유지)
            critic (nn.Module): 가치 네트워크 (Critic)  # (기존 주석 유지)
            writer (SummaryWriter, optional): TensorBoard 기록용 writer  # (기존 주석 유지)
        """
        self.device = torch.device(
            config_manager.get_device()
        )  # config에서 device(cpu/cuda)를 가져와 torch device로 설정
        self.actor = actor.to(self.device)  # actor 모델을 지정 device로 이동
        self.critic = critic.to(self.device)  # critic 모델을 지정 device로 이동

        self.optimizer_actor = optim.Adam(
            self.actor.parameters(), lr=config_manager.get_learning_rate() * 0.5
        )  # actor용 Adam 옵티마이저(학습률을 critic보다 낮게 0.5배)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=config_manager.get_learning_rate()
        )  # critic용 Adam 옵티마이저(기본 학습률)

        self.gamma = config_manager.get_gamma()  # 할인율(미래 보상 현재가치 반영 비율)
        self.clampepsilon = (
            config_manager.get_clampepsilon()
        )  # PPO 클리핑 범위(epsilon, clip param)
        self.batch_size = config_manager.get_batch_size()  # 미니배치 크기
        self.entropy_coef = (
            config_manager.get_entropy_coef()
        )  # 엔트로피 보상 계수(탐험 유도)
        self.temperature = (
            config_manager.get_temperature()
        )  # softmax temperature(분포를 평평/뾰족하게 조절)

        self.writer = writer  # TensorBoard 기록 객체 저장
        self.train_step = (
            0  # 학습/로깅 스텝 카운터(디버깅 및 epsilon decay 등에도 사용)
        )
        self.epsilon = config_manager.get_epsilon()  # epsilon-greedy 탐험 확률(초기값)
        self.epsilon_min = (
            config_manager.get_epsilon_min()
        )  # epsilon 최소값(더 이상 줄어들지 않게)
        self.epsilon_decay = (
            config_manager.get_epsilon_decay()
        )  # epsilon 감쇠율(스텝마다 곱해 감소)
        self.action_dim = actor.action_dim  # 액션 공간 크기(가능한 행동 개수)

    def select_action(
        self, state
    ):  # 현재 상태에서 행동(action)과 log_prob, value를 선택/계산
        """
        현재 상태에서 액션을 선택하고 로그 확률 및 상태 가치 반환  # (기존 주석 유지)

        Args:
            state (np.ndarray): 환경으로부터 받은 현재 상태 (shape: [seq_len, feature_dim])  # (기존 주석 유지)

        Returns:
            tuple: (action (int), log_prob (float), value (float))  # (기존 주석 유지)
        """
        """현재 상태에서 액터를 통해 행동을 선택하고 크리틱의 상태 가치를 함께 반환"""  # (기존 주석 유지)
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )  # (1) numpy->tensor 변환 후 배치 차원(1) 추가하고 device로 이동

        # 🔥 보유 주식수 feature 확인 로그 (1000 스텝마다)  # (기존 주석 유지)
        if (
            self.train_step % 1000 == 0
        ):  # train_step이 1000의 배수일 때만 로그 출력(과도한 로그 방지)
            log_manager.logger.debug(  # 디버그 레벨 로그 출력
                f"[Agent] State 확인:\n"  # 로그 문자열(상태 확인 헤더)
                f"  - State shape: {state.shape}\n"  # 원본 state의 shape 출력
                f"  - State tensor shape: {state_tensor.shape}\n"  # 텐서 변환 후 shape 출력(배치 차원 포함)
                f"  - 마지막 feature (보유 주식수): {state[:, -1]}\n"  # feature의 마지막 컬럼을 보유주식수로 가정하여 출력
                f"  - 보유 주식수 평균: {state[:, -1].mean():.4f}\n"  # 보유주식수 평균 출력
                f"  - 보유 주식수 최대: {state[:, -1].max():.4f}\n"  # 보유주식수 최대 출력
                f"  - 보유 주식수 최소: {state[:, -1].min():.4f}"  # 보유주식수 최소 출력
            )

        if not torch.isfinite(
            state_tensor
        ).all():  # state_tensor에 inf/nan이 하나라도 있는지 검사
            print(
                "⚠️ Invalid state detected:", state_tensor
            )  # 비정상 상태 텐서를 콘솔에 경고 출력

        logits = self.actor(
            state_tensor
        )  # actor가 상태를 입력받아 행동 logits(정규화 전 점수) 출력
        probs = torch.softmax(
            logits / self.temperature, dim=-1
        )  # temperature로 스케일링 후 softmax로 확률 분포로 변환
        dist = Categorical(
            probs
        )  # 확률분포(probs)로 카테고리 분포 객체 생성(샘플링/로그확률/엔트로피 계산용)

        # 🔥 탐험 강화: epsilon-greedy + softmax sampling 혼합  # (기존 주석 유지)
        if random.random() < self.epsilon:  # 0~1 난수 < epsilon이면 탐험(랜덤) 선택
            # 완전 랜덤 탐험  # (기존 주석 유지)
            action = random.choice(
                range(self.action_dim)
            )  # action_dim 범위에서 랜덤 action 선택
        else:  # 그렇지 않으면(1-epsilon 확률) 정책 기반 선택
            # 정책 기반 선택 (softmax sampling)  # (기존 주석 유지)
            action = (
                dist.sample().item()
            )  # Categorical 분포에서 샘플링(확률에 비례)하여 action 선택 후 파이썬 int로 변환

        log_prob = dist.log_prob(
            torch.tensor(action).to(self.device)
        )  # 선택한 action의 로그확률 계산(정책 gradient/PPO ratio 계산에 사용)

        # 🔥 Epsilon 감소를 더 천천히  # (기존 주석 유지)
        self.epsilon = max(
            self.epsilon * self.epsilon_decay, self.epsilon_min
        )  # epsilon을 감쇠시키되 최소값 이하로는 내려가지 않도록 제한

        # ✅ TensorBoard 기록  # (기존 주석 유지)
        if (
            self.writer and self.action_dim <= 10
        ):  # writer가 있고 액션이 너무 많지 않을 때만(그래프 난잡 방지)
            for i in range(self.action_dim):  # 각 action별 확률을 기록하기 위한 반복
                self.writer.add_scalar(
                    f"Action_Prob/Action_{i}", probs[0, i].item(), self.train_step
                )  # action i 확률을 스칼라로 기록(배치 0)

        # ✅ 디버깅 로그 (더 자주 출력)  # (기존 주석 유지)
        if (
            self.train_step % 100 == 0
        ):  # 500 → 100으로 변경  # (기존 주석 유지) 100스텝마다 디버그 로그 출력
            topk = sorted(
                enumerate(probs[0].tolist()), key=lambda x: x[1], reverse=True
            )[
                :3
            ]  # 확률 상위 3개 action을 (idx, prob) 형태로 정렬/추출
            topk_log = {}  # 상위 3개를 사람이 읽기 좋게 정리할 dict
            for idx, val in topk:  # top-3 (action index, probability) 순회
                if idx == 0:  # idx가 0이면(사용자 정의 매핑) 관망
                    action_type = "관망"  # 행동 타입 라벨
                    action_info = f"{action_type}"  # 로그에 넣을 설명 문자열
                elif idx == 1:  # idx가 1이면(사용자 정의 매핑) 전부매수
                    action_type = "전부매수"  # 행동 타입 라벨
                    action_info = f"{action_type}"  # 로그에 넣을 설명 문자열
                else:  # idx == 2  # (기존 주석 유지) 나머지(여기서는 2로 가정)는 전부매도
                    action_type = "전부매도"  # 행동 타입 라벨
                    action_info = f"{action_type}"  # 로그에 넣을 설명 문자열
                topk_log[f"Action_{idx}"] = (
                    f"{val:.4f} → {action_info}"  # 확률과 행동 설명을 묶어 기록
                )
            log_manager.logger.debug(
                f"[Step {self.train_step}] Epsilon: {self.epsilon:.3f}, Top-3 probs:\n{topk_log}"
            )  # 현재 스텝/epsilon/top-3를 디버그 로그로 출력

        self.train_step += 1  # 액션 선택이 끝났으니 스텝 카운터 증가(다음 호출 대비)
        value = self.critic(
            state_tensor
        ).item()  # critic이 상태가치 V(s)를 예측하고 스칼라 값으로 변환
        return (
            action,
            log_prob.item(),
            value,
        )  # action, log_prob(파이썬 float), value(파이썬 float) 반환

    def update(self, memory):  # 저장된 memory로 PPO 업데이트 수행
        """
        저장된 에피소드 메모리를 기반으로 정책 및 가치 함수 업데이트  # (기존 주석 유지)

        Args:
            memory (list): (state, action, reward, log_prob, value) 튜플의 리스트  # (기존 주석 유지)
        """
        """메모리로부터 Advantage 기반 Actor-Critic 업데이트 수행"""  # (기존 주석 유지)
        states, actions, rewards, old_log_probs, values = zip(
            *memory
        )  # memory list를 항목별 튜플로 분해(언패킹)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.device
        )  # states를 numpy->tensor로 변환 후 device로 이동
        actions = torch.tensor(actions).to(
            self.device
        )  # actions를 tensor로 변환 후 device로 이동
        rewards = torch.tensor(rewards).to(
            self.device
        )  # rewards를 tensor로 변환 후 device로 이동
        old_log_probs = torch.tensor(old_log_probs).to(
            self.device
        )  # 이전 정책의 log_prob를 tensor로 변환 후 device로 이동(PPO ratio 기준)
        values = torch.tensor(values).to(
            self.device
        )  # 이전 critic value 예측값을 tensor로 변환 후 device로 이동(advantage 계산)

        # Discounted rewards 계산  # (기존 주석 유지)
        discounted_rewards = []  # 각 시점의 누적 할인 보상(returns)을 저장할 리스트
        running_add = 0  # 뒤에서부터 누적할 변수(미래부터 거꾸로 누적)
        for r in reversed(
            rewards
        ):  # rewards를 뒤에서부터 순회(리턴 계산의 전형적인 방식)
            running_add = r + self.gamma * running_add  # G_t = r_t + gamma * G_{t+1}
            discounted_rewards.insert(
                0, running_add
            )  # 계산된 값을 앞에 삽입해 원래 순서로 맞춤
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(
            self.device
        )  # list -> tensor로 변환 후 device로 이동

        # Advantage 계산  # (기존 주석 유지)
        advantages = (
            discounted_rewards - values
        )  # critic이 예측한 values와 실제 return의 차이  # (기존 주석 유지) advantage = 실제 리턴 - 가치추정
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )  # advantage 정규화(학습 안정화, 스케일 문제 완화)

        for i in range(
            0, len(states), self.batch_size
        ):  # 전체 데이터를 batch_size 단위로 미니배치 학습
            b_states = states[i : i + self.batch_size]  # 현재 미니배치 states 슬라이싱
            b_actions = actions[
                i : i + self.batch_size
            ]  # 현재 미니배치 actions 슬라이싱
            b_advantages = advantages[
                i : i + self.batch_size
            ].detach()  # 현재 미니배치 advantages(그래프 분리: actor 업데이트에서 target로 사용)
            b_returns = discounted_rewards[
                i : i + self.batch_size
            ].detach()  # 현재 미니배치 returns(critic 타깃, 그래프 분리)
            b_old_log_probs = old_log_probs[
                i : i + self.batch_size
            ].detach()  # 현재 미니배치 old log_probs(그래프 분리)

            # Actor 업데이트  # (기존 주석 유지)
            logits = self.actor(b_states)  # 현재 actor로부터 logits 계산
            probs = torch.softmax(
                logits / self.temperature, dim=-1
            )  # temperature 적용 후 softmax로 확률 계산
            dist = Categorical(probs)  # 확률로 카테고리 분포 생성
            new_log_probs = dist.log_prob(
                b_actions
            )  # 현재 정책에서 선택된 action들의 log_prob 계산

            ratio = torch.exp(
                new_log_probs - b_old_log_probs
            )  # PPO ratio = pi(a|s) / pi_old(a|s) = exp(logpi - logpi_old)
            clipped_ratio = torch.clamp(
                ratio, 1 - self.clampepsilon, 1 + self.clampepsilon
            )  # ratio를 [1-eps, 1+eps]로 클리핑

            actor_loss = -torch.min(
                ratio * b_advantages, clipped_ratio * b_advantages
            ).mean()  # actor는 advantage 값을 기준으로 확률을 조정  # (기존 주석 유지) PPO surrogate objective(최소값) 음수로 손실화
            actor_loss -= (
                self.entropy_coef * dist.entropy().mean()
            )  # 엔트로피 항을 빼서(손실을 낮추려면 엔트로피가 커지게) 탐험 유도

            # Critic 업데이트  # (기존 주석 유지)
            values_pred = self.critic(
                b_states
            ).squeeze()  # 현재 상태에서 critic이 예측한 값  # (기존 주석 유지) critic의 V(s) 예측값
            critic_loss = nn.MSELoss()(
                values_pred, b_returns
            )  # 상태의 실제 보상 누적값과 자신이 예측한  V(s)의 차이를 줄이도록 학습  # (기존 주석 유지) MSE로 value 회귀

            # Actor 업데이트  # (기존 주석 유지)
            self.optimizer_actor.zero_grad()  # actor optimizer의 기울기 초기화
            actor_loss.backward()  # actor 손실에 대한 역전파로 grad 계산
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=1.0
            )  # actor 파라미터 gradient 클리핑(폭주 방지)
            self.optimizer_actor.step()  # actor 파라미터 업데이트(optimizer step)

            # Critic 업데이트  # (기존 주석 유지)
            self.optimizer_critic.zero_grad()  # critic optimizer의 기울기 초기화
            critic_loss.backward()  # critic 손실에 대한 역전파로 grad 계산
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=1.0
            )  # critic 파라미터 gradient 클리핑(폭주 방지)
            self.optimizer_critic.step()  # critic 파라미터 업데이트(optimizer step)

            # TensorBoard 기록  # (기존 주석 유지)
            if self.writer:  # writer가 있으면 TensorBoard에 메트릭 기록
                self.writer.add_scalar(
                    "Loss/Actor", actor_loss.item(), self.train_step
                )  # actor loss 기록
                self.writer.add_scalar(
                    "Loss/Critic", critic_loss.item(), self.train_step
                )  # critic loss 기록
                self.writer.add_scalar(
                    "Advantage/Mean", b_advantages.mean().item(), self.train_step
                )  # advantage 평균 기록(학습 상태 확인)
                self.writer.add_scalar(
                    "Returns/Mean", b_returns.mean().item(), self.train_step
                )  # returns 평균 기록(보상 스케일 확인)

            self.train_step += (
                1  # 미니배치 업데이트가 끝났으니 train_step 증가(로깅/감쇠 등에 영향)
            )
