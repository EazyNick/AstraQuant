from env.stock_env import StockTradingEnv
from models.transformer_model import StockTransformer
from agents.ppo_agent import PPOAgent
import numpy as np

def train_agent(env, agent, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        memory = []
        total_reward = 0

        for t in range(len(env.stock_data) - 30):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward))
            state = next_state
            total_reward += reward

            if done:
                break

        agent.update(memory)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    stock_prices = np.linspace(100, 200, num=1000)  # 샘플 데이터
    env = StockTradingEnv(stock_prices)
    model = StockTransformer()
    agent = PPOAgent(model)
    train_agent(env, agent, episodes=100)
