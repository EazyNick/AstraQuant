import os

### 사용법 ###
"""  
import os
import sys

current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..")) # 현재 디렉토리에 따라 이 부분 수정
sys.path.append(project_root)

from manage import PathManager
path_manager = PathManager()

# 원하는 경로 추가
sys.path.append(path_manager.get_path("logs"))
sys.path.append(path_manager.get_path("assets_stagingarea"))
sys.path.append(path_manager.get_path("module"))
sys.path.append(path_manager.get_path("display"))

# import
try:
    from logs import log_manager
    from module import ProcessStep
    from display import screenhandler
except Exception as e:
    log_manager.logger.debug("This is a debug message for testing purposes.")(f"임포트 실패: {e}")
"""


class PathManager:
    """
    프로젝트 내 모든 주요 폴더 경로를 관리하는 클래스.
    """
    def __init__(self):
        # 프로젝트 최상위 디렉토리 경로 계산
        self.project_root = os.path.abspath(os.path.join(__file__, "..", ".."))

        # 주요 폴더 경로 설정
        self.folders = { 
            "agents": os.path.join(self.project_root, "agents"),
            "config": os.path.join(self.project_root, "config"),
            "data": os.path.join(self.project_root, "data"),
            "env": os.path.join(self.project_root, "env"),
            "git": os.path.join(self.project_root, "git"),
            "logs": os.path.join(self.project_root, "logs"),
            "manage": os.path.join(self.project_root, "manage"),
            "models": os.path.join(self.project_root, "models"),
            "output": os.path.join(self.project_root, "output"),
            "training": os.path.join(self.project_root, "training"),
        }

    def get_path(self, folder_name):
        """
        지정된 폴더 이름의 경로를 반환합니다.
        
        :param folder_name: 폴더 이름 (예: "utils", "assets_temp")
        :return: 해당 폴더의 절대 경로
        """
        if folder_name not in self.folders:
            raise ValueError(f"'{folder_name}'은(는) 관리되지 않는 폴더입니다.")
        return self.folders[folder_name]

    def list_all_paths(self):
        """
        관리 중인 모든 폴더의 경로를 반환합니다.
        :return: dict
        """
        return self.folders

if __name__ == "__main__":
    path_manager = PathManager()

    print("=== 관리 중인 디렉토리 경로 목록 ===")
    all_paths = path_manager.list_all_paths()
    for folder_name, path in all_paths.items():
        print(f"{folder_name}: {path}")
