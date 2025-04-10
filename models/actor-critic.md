🟩 환경 (StockTradingEnv)
    │
    ▼
(1) 현재 상태(state): shape = (batch, seq_len, input_dim + 1)  ← 보유 주식 수 포함
    │
    ├──▶ 🟠 Actor 네트워크
    │         └─ (2) Embedding → (3) Positional Encoding → (4) Transformer 인코더
    │         └─ (5) 마지막 타임스텝 추출 (x[:, -1, :])
    │         └─ (6) FC → (7) 액션 logits → softmax → 확률분포 π(a|s)
    │              ↓
    │         샘플링 or ε-greedy → 선택된 action, log_prob
    │
    └──▶ 🔵 Critic 네트워크
              └─ 동일 구조 (Embedding → PE → Transformer)
              └─ (6) FC → (7) 상태 가치 V(s)
    
          ↓
📦 환경에 action 전달 (env.step(action))
    └─ next_state, reward, done 반환

          ↓
🧠 메모리에 (state, action, reward, log_prob, value) 저장

          ↓
💡 일정 step마다 업데이트 수행:
    ├─ (1) reward로 discounted return 계산
    ├─ (2) advantage = return - value
    ├─ (3) actor_loss = PPO objective (ratio, clipped)
    ├─ (4) critic_loss = MSE(value_pred, return)
    └─ (5) 두 네트워크 모두 gradient descent로 학습

📈 TensorBoard 기록: 확률, 로스, advantage, return

"""
(1) 입력 데이터 (batch, seq_len, input_dim)
        │
        ▼
(2) Embedding (Linear 변환 → model_dim 크기로 맞춤)
        │
        ▼
(3) Positional Encoding 추가 (시간 정보 반영)
        │
        ▼
(4) Transformer 인코더 (TransformerEncoder가 `num_layers`개 쌓인)
        │
        ▼
(5) 마지막 타임스텝의 출력을 가져옴 (x[:, -1, :])
        │
        ▼
(6) Fully Connected Layer (3차원: Buy, Hold, Sell)
        │
        ▼
(7) 최종 예측 결과 (확률 값)
"""