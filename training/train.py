import torch
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
sys.path.append(path_manager.get_path("env"))
sys.path.append(path_manager.get_path("models"))
sys.path.append(path_manager.get_path("agents"))
sys.path.append(path_manager.get_path("data"))

try:
    from logs import log_manager
    from config import config_manager
    from env.stock_env import StockTradingEnv
    from models.transformer_model import StockTransformer
    from agents.ppo_agent import PPOAgent
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

    def save_checkpoint(self, model, optimizer, episode):
        """
        체크포인트를 저장하는 함수 (모델 + 옵티마이저 + 현재 진행된 에피소드 포함)

        Args:
            model (torch.nn.Module): 저장할 모델
            optimizer (torch.optim.Optimizer): 옵티마이저 상태
            episode (int): 현재 학습 진행된 에피소드
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'episode': episode
        }
        torch.save(checkpoint, self.checkpoint_path)
        log_manager.logger.info(f"✅ 체크포인트 저장 완료: {self.checkpoint_path} (Episode {episode})")

    def load_checkpoint(self, model, optimizer):
        """
        체크포인트를 로드하는 함수

        Args:
            model (torch.nn.Module): 불러올 모델
            optimizer (torch.optim.Optimizer): 불러올 옵티마이저 상태

        Returns:
            int: 마지막 학습된 에피소드 번호 (없으면 0 반환)
        """
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            episode = checkpoint['episode']
            log_manager.logger.info(f"✅ 체크포인트 로드 완료: {self.checkpoint_path} (Episode {episode})")
            return episode
        else:
            log_manager.logger.info("⚠️ 체크포인트가 존재하지 않습니다. 새로운 학습을 시작합니다.")
            return 0

def train_agent(env, agent, episodes, training_manager):
    """ PPO 에이전트를 학습시키는 함수 """
    log_manager.logger.info(f"🎯 학습 시작")

    # 체크포인트 로드 (이전 학습 기록이 있으면 이어서 시작)
    start_episode = training_manager.load_checkpoint(agent.model, agent.optimizer)
    best_reward = float('-inf')  # 최고 리워드 기록 초기화
    saveflag = False

    for episode in range(start_episode, episodes):
        state = env.reset()
        memory = []
        total_reward = 0

        for t in range(len(env.stock_data) - config_manager.get_observation_window()):
            action = agent.select_action(state)  # PPOAgent가 액션 선택
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward))
            state = next_state
            total_reward += reward

            if done or len(memory) >= agent.batch_size:
                agent.update(memory)  # PPO 업데이트 수행
                memory = []  # 배치 학습 후 메모리 초기화

        final_portfolio_value = env.balance + (env.shares_held * env.stock_data[env.current_step, 0])
        log_manager.logger.debug(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, final_portfolio_value: {final_portfolio_value:.2f}")

        # 매 100번째 에피소드마다 모델과 체크포인트 저장
        if (episode + 1) % 100 == 0:
            training_manager.save_model(agent.model, episode=(episode + 1))
            training_manager.save_checkpoint(agent.model, agent.optimizer, episode+1)  # 체크포인트 저장
            saveflag = True
            log_manager.logger.info(f"✅ 체크포인트 및 모델 저장 완료 (Episode {episode+1})")

         # 현재 에피소드의 보상이 최고 보상(best_reward)보다 높을 경우 저장
        if saveflag == False and total_reward > best_reward:
            best_reward = total_reward  # 최고 리워드 갱신
            training_manager.save_model(agent.model, episode=(episode + 1))
            training_manager.save_checkpoint(agent.model, agent.optimizer, episode + 1)  # 체크포인트 저장
            log_manager.logger.info(f"✅ 최고 리워드 갱신! 모델 저장 완료 (Episode {episode+1})")

    # 최종 학습 완료 후 모델 저장
    training_manager.save_model(agent.model)
    log_manager.logger.info(f"✅ 최종 모델 저장 완료: {training_manager.save_path}")

if __name__ == "__main__":
    # ✅ TrainingManager 인스턴스 생성
    training_manager = TrainingManager()

    # ✅ 데이터 로드
    data, input_dim = load_stock_data("data/csv/sp500_training_data.csv")

    # ✅ 환경 및 모델 생성
    env = StockTradingEnv(data)
    model = StockTransformer(input_dim=input_dim)
    agent = PPOAgent(model)

    # ✅ 학습 시작
    train_agent(env, agent, episodes=config_manager.get_episodes(), training_manager=training_manager)
