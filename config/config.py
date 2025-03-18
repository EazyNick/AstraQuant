import yaml
import torch
import os
import sys

current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..")) # 현재 디렉토리에 따라 이 부분 수정
sys.path.append(project_root)

from manage import PathManager
path_manager = PathManager()

# 원하는 경로 추가
sys.path.append(path_manager.get_path("logs"))

try:
    from logs import log_manager
except Exception as e:
    print(f"임포트 실패: {e}")

class ConfigManager:
    """ config.yaml을 관리하는 Singleton 클래스 (Getter & Setter 포함) """

    _instance = None  # 싱글턴 인스턴스 저장

    def __new__(cls, config_filename="config.yaml"):
        """ 싱글턴 패턴 적용: 인스턴스가 이미 존재하면 기존 인스턴스를 반환 """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config(config_filename)
        return cls._instance

    def _init_config(self, config_filename):
        """ 설정 파일을 로드하고 GPU/CPU 설정을 자동으로 조정 """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, config_filename)

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file '{config_filename}' not found in {self.script_dir}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # GPU 설정 조정 (한 번만 수행)
        self._adjust_device()

    def _adjust_device(self):
        """ GPU 사용 여부를 확인하고 없으면 자동으로 CPU로 변경 (한 번만 설정됨) """
        if hasattr(self, "device_set") and self.device_set:
            return  # 이미 device 설정이 한 번 수행되었으면 다시 변경하지 않음

        requested_device = self.config["general"].get("device", "cuda")  # 기본값은 "cuda"
        
        if requested_device == "cuda" and not torch.cuda.is_available():
            log_manager.logger.debug("⚠️ CUDA GPU를 찾을 수 없습니다. CPU로 변경합니다.")
            requested_device = "cpu"
        
        if requested_device not in ["cuda", "cpu"]:
            raise RuntimeError("❌ 사용 가능한 device가 없습니다. (cuda 또는 cpu 중 하나를 사용하세요.)")

        self.config["general"]["device"] = requested_device  # 업데이트된 디바이스 저장
        self.device_set = True  # device 설정이 한 번만 수행되도록 플래그 추가

    ## =============================== ##
    ##        General 설정 Getter       ##
    ## =============================== ##
    def get_device(self):
        return self.config["general"]["device"]

    def get_seed(self):
        return self.config["general"]["seed"]

    ## =============================== ##
    ##        Env 설정 Getter          ##
    ## =============================== ##
    def get_initial_balance(self):
        return self.config["env"]["initial_balance"]

    def get_observation_window(self):
        return self.config["env"]["observation_window"]
    

    def get_transaction_fee(self):
        return self.config["env"].get("transaction_fee", 0.001)  # 기본값 설정

    ## =============================== ##
    ##        Training 설정 Getter     ##
    ## =============================== ##
    def get_episodes(self):
        return self.config["training"]["episodes"]

    def get_batch_size(self):
        return self.config["training"]["batch_size"]

    def get_learning_rate(self):
        return self.config["training"]["learning_rate"]

    def get_gamma(self):
        return self.config["training"]["gamma"]

    def get_clampepsilon(self):
        return self.config["training"]["clampepsilon"]

    def get_num_workers(self):
        return self.config["training"]["num_workers"]

    ## =============================== ##
    ##        Model 설정 Getter        ##
    ## =============================== ##
    def get_input_dim(self):
        return self.config["model"]["input_dim"]

    def get_model_dim(self):
        return self.config["model"]["model_dim"]

    def get_num_heads(self):
        return self.config["model"]["num_heads"]

    def get_num_layers(self):
        return self.config["model"]["num_layers"]

    ## =============================== ##
    ##           Setter Methods        ##
    ## =============================== ##
    def set_seed(self, value):
        self.config["general"]["seed"] = value

    def set_initial_balance(self, value):
        self.config["env"]["initial_balance"] = value

    def set_observation_window(self, value):
        self.config["env"]["observation_window"] = value

    def set_episodes(self, value):
        self.config["training"]["episodes"] = value

    def set_batch_size(self, value):
        self.config["training"]["batch_size"] = value

    def set_learning_rate(self, value):
        self.config["training"]["learning_rate"] = value

    def set_gamma(self, value):
        self.config["training"]["gamma"] = value

    def set_clampepsilon(self, value):
        self.config["training"]["clampepsilon"] = value

    def set_num_workers(self, value):
        self.config["training"]["num_workers"] = value

    def set_input_dim(self, value):
        self.config["model"]["input_dim"] = value

    def set_model_dim(self, value):
        self.config["model"]["model_dim"] = value

    def set_num_heads(self, value):
        self.config["model"]["num_heads"] = value

    def set_num_layers(self, value):
        self.config["model"]["num_layers"] = value

    def set_transaction_fee(self, value):
        self.config["env"]["transaction_fee"] = value

    def save_config(self):
        """ 변경된 설정을 config.yaml 파일에 저장 """
        with open(self.config_path, "w", encoding="utf-8") as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
        log_manager.logger.debug("✅ 설정이 config.yaml 파일에 저장되었습니다.")

    def show_config(self):
        """ 현재 설정값 출력 """
        log_manager.logger.debug("🔹 현재 설정값:")
        for category, params in self.config.items():
            log_manager.logger.debug(f"[{category}]")
            for key, value in params.items():
                log_manager.logger.debug(f"  {key}: {value}")
            log_manager.logger.debug("")

# 예제 실행
if __name__ == "__main__":
    config_manager = ConfigManager()
    
    # 현재 설정 출력
    config_manager.show_config()

    # 특정 설정 값 가져오기
    log_manager.logger.debug(f"🎯 학습 장치: {config_manager.get_device()}")
    log_manager.logger.debug(f"🎯 학습 에피소드 수: {config_manager.get_episodes()}")

    # 특정 설정 변경
    config_manager.set_episodes(10)
    config_manager.set_learning_rate(0.0005)

    # 변경된 설정 확인
    config_manager.show_config()

    # 설정 파일 저장
    # config_manager.save_config()
