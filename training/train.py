import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter

current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..")) # 현재 디렉토리에 따라 이 부분 수정
sys.path.append(project_root)

from manage import PathManager
path_manager = PathManager()

# 원하는 경로 추가
sys.path.append(path_manager.get_path("config"))
sys.path.append(path_manager.get_path("logs"))
sys.path.append(path_manager.get_path("env"))
sys.path.append(path_manager.get_path("models"))
sys.path.append(path_manager.get_path("agents"))
sys.path.append(path_manager.get_path("data"))

try:
    from logs import log_manager
    from config import config_manager
    from env.stock_env import StockTradingEnv
    from models.actor_network import ActorNetwork
    from models.critic_network import CriticNetwork
    from agents.actor_critic_agent import ActorCriticAgent
    from data.data_loader import load_stock_data
    from config import config_manager
except Exception as e:
    print(f"임포트 실패: {e}")

class TrainingManager:
    """ 학습된 모델을 저장 및 관리하는 클래스 """
    def __init__(self, directory=None, filename="ppo_stock_trader.pth", checkpoint_filename="ppo_checkpoint.pth"):
        """
        TrainingManager 초기화

        Args:
            directory (str, optional): 모델 저장 디렉토리. 기본값은 프로젝트 루트 디렉토리에서 `output` 폴더.
            filename (str): 저장할 모델 파일 이름.
            checkpoint_filename (str): 체크포인트 파일 이름.
        """
        if not hasattr(self, 'initialized'):  # 인스턴스가 초기화되었는지 확인
            default_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
            self.directory = directory or default_directory  # ✅ 사용자가 지정한 경로가 없으면 기본값 사용
            self.filename = filename
            self.checkpoint_filename = checkpoint_filename
            self.save_path = os.path.join(self.directory, self.filename)
            self.checkpoint_path = os.path.join(self.directory, self.checkpoint_filename)
            self.epsilon = config_manager.get_epsilon()
            log_manager.logger.debug(f"✅ 모델 저장 경로: {self.save_path}")

            os.makedirs(self.directory, exist_ok=True)  # 폴더가 없으면 자동 생성
            self.initialized = True

    def save_model(self, model, episode=None):
        """
        모델 가중치를 저장하는 함수 (전체 모델이 아니라 가중치만 저장)

        Args:
            model (torch.nn.Module): 저장할 모델
            episode (int, optional): 에피소드 번호를 포함하여 저장 (기본값: None)
        """
        if episode is not None:
            filename = f"ppo_stock_trader_episode_{episode}.pth"  # ✅ 에피소드 번호 포함
        else:
            filename = self.filename

        save_path = os.path.join(self.directory, filename)

        try:
            torch.save(model.state_dict(), save_path)  # 🔥 가중치만 저장
            log_manager.logger.info(f"✅ 모델 저장 완료: {save_path}")
        except Exception as e:
            log_manager.logger.error(f"❌ 모델 저장 실패: {e}")

    def save_checkpoint(self, actor, critic, optimizer_actor, optimizer_critic, episode):
        """
        체크포인트를 저장하는 함수 (Actor-Critic)

        Args:
            actor (torch.nn.Module): Actor 모델
            critic (torch.nn.Module): Critic 모델
            optimizer_actor (torch.optim.Optimizer): Actor 옵티마이저
            optimizer_critic (torch.optim.Optimizer): Critic 옵티마이저
            episode (int): 저장할 시점의 에피소드 번호
        """
        checkpoint = {
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'optimizer_actor': optimizer_actor.state_dict(),
            'optimizer_critic': optimizer_critic.state_dict(),
            'episode': episode,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, self.checkpoint_path)
        log_manager.logger.info(f"✅ 체크포인트 저장 완료: {self.checkpoint_path} (Episode {episode}, epsilon: {self.epsilon:.6f})")

    def load_checkpoint(self, actor, critic, optimizer_actor, optimizer_critic, agent=None):
        """
        체크포인트를 로드하는 함수

        Args:
            actor (torch.nn.Module): Actor 모델
            critic (torch.nn.Module): Critic 모델
            optimizer_actor (torch.optim.Optimizer): Actor 옵티마이저
            optimizer_critic (torch.optim.Optimizer): Critic 옵티마이저
            agent (ActorCriticAgent, optional): 에이전트 인스턴스 (epsilon 복원용)

        Returns:
            int: 마지막 저장된 에피소드 번호. 체크포인트가 없으면 0 반환.
        """
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic.load_state_dict(checkpoint['critic_state_dict'])
            optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
            optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
            episode = checkpoint['episode']
            self.epsilon = checkpoint.get('epsilon', self.epsilon)

            if agent is not None:
                agent.epsilon = self.epsilon  # ✅ PPOAgent의 epsilon도 업데이트

            return episode
        else:
            log_manager.logger.info("⚠️ 체크포인트가 존재하지 않습니다. 새로운 학습을 시작합니다.")
            return 0

def train_agent(env, agent, episodes, training_manager):
    """ PPO 에이전트를 학습시키는 함수 
    
    Args:
        env (StockTradingEnv): 주식 거래 환경 인스턴스
        agent (ActorCriticAgent): 학습할 에이전트
        episodes (int): 총 학습 에피소드 수
        training_manager (TrainingManager): 모델 저장 및 체크포인트 관리 클래스
    """
    log_manager.logger.info(f"🎯 학습 시작")

    # 체크포인트 로드 (이전 학습 기록이 있으면 이어서 시작)
    start_episode = training_manager.load_checkpoint(agent.actor, agent.critic, agent.optimizer_actor, agent.optimizer_critic, agent)
    best_reward = float('-inf')  # 최고 리워드 기록 초기화

    for episode in range(start_episode, episodes):
        state = env.reset()
        memory = []
        total_reward = 0

        for t in range(len(env.stock_data) - config_manager.get_observation_window()):
            # balance = env.balance
            # shares_held = env.shares_held
            # current_price = env.stock_data[env.current_step, 0]

            # ✅ PPOAgent에게 환경 정보를 전달하여 액션 선택
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, log_prob, value))  # 메모리에 실제 액션 저장
            state = next_state
            total_reward += reward

            if done or len(memory) >= agent.batch_size:
                agent.update(memory)  # PPO 업데이트 수행
                memory = []  # 배치 학습 후 메모리 초기화

        final_portfolio_value = env.balance + (env.shares_held * env.stock_data[env.current_step, 0])
        log_manager.logger.info(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, final_portfolio_value: {final_portfolio_value:.2f}")

        # 매 100번째 에피소드마다 모델과 체크포인트 저장
        if (episode + 1) % 50 == 0:
            training_manager.save_model(agent.actor, episode=(episode + 1))
            training_manager.save_checkpoint(agent.actor, agent.critic, agent.optimizer_actor, agent.optimizer_critic, episode + 1)  # 체크포인트 저장
            log_manager.logger.info(f"✅ 체크포인트 및 모델 저장 완료 (Episode {episode+1})")

         # 현재 에피소드의 보상이 최고 보상(best_reward)보다 높을 경우 저장
        if total_reward > best_reward:
            best_reward = total_reward  # 최고 리워드 갱신
            training_manager.save_model(agent.actor, episode=(episode + 1))
            training_manager.save_checkpoint(agent.actor, agent.critic, agent.optimizer_actor, agent.optimizer_critic, episode + 1)  # 체크포인트 저장
            log_manager.logger.info(f"✅ 최고 리워드 갱신! 모델 저장 완료 (Episode {episode+1})")

    # 최종 학습 완료 후 모델 저장
    training_manager.save_model(agent.actor)
    log_manager.logger.info(f"✅ 최종 모델 저장 완료: {training_manager.save_path}")

if __name__ == "__main__":
    # ✅ TrainingManager 인스턴스 생성
    training_manager = TrainingManager()

    # ✅ 데이터 로드
    data, input_dim = load_stock_data("data/csv/005930.KS_combined_train_data.csv")

    # ✅ 텐서보드 writer 생성
    writer = SummaryWriter(log_dir="logs/training")

    # ✅ 환경 및 모델 생성
    env = StockTradingEnv(data, writer=writer)
    actor = ActorNetwork(input_dim=input_dim)
    critic = CriticNetwork(input_dim=input_dim)
    agent = ActorCriticAgent(actor, critic, writer=writer)

    # ✅ 학습 시작
    train_agent(env, agent, episodes=config_manager.get_episodes(), training_manager=training_manager)

