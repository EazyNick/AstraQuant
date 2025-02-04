import yaml
import os

class DeployConfig:
    """
    YAML 설정 파일을 불러와 저장하는 클래스
    """
    def __init__(self, config_path="git\gitconfig.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일이 존재하지 않습니다: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def get(self, key):
        """
        설정값 반환 메서드
        Args:
            key (str): 설정 파일의 키값

        Returns:
            str: 설정값
        """
        return self.config.get(key, None)
