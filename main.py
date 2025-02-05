# conda activate AstraQuant

from env.stock_env import StockTradingEnv
from models.transformer_model import StockTransformer
from agents.ppo_agent import PPOAgent
from training.train import train_agent
from data.data_loader import load_stock_data

stock_prices = load_stock_data("data/csv/sp500_training_data.csv")

env = StockTradingEnv(stock_prices)
model = StockTransformer()
agent = PPOAgent(model)

train_agent(env, agent, episodes=1)
