from .config import ConfigManager  # 싱글턴 ConfigManager 불러오기

# 싱글턴 인스턴스를 생성하여 모듈 내에서 공유
config_manager = ConfigManager()
