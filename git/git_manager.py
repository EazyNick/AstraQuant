import os
import sys
import subprocess
from functools import cached_property
from .config_loader import DeployConfig 

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
sys.path.append(project_root)

from manage import PathManager

path_manager = PathManager()
sys.path.append(path_manager.get_path("utils"))
sys.path.append(path_manager.get_path("logs"))
sys.path.append(path_manager.get_path("common"))

from logs import log_manager

class GitSyncError(Exception):
    """Git 동기화 중 발생하는 예외 클래스"""
    pass

class GitManager:
    """
    GitHub 레포지토리 동기화를 관리하는 싱글턴 클래스
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GitManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 설정값 불러오기
        self.config = DeployConfig()
        self.repository = self.config.get("repository")
        self.branch = self.config.get("branch")

    @cached_property
    def git_executable(self):
        """
        Git 실행 파일 경로를 반환
        """
        return self.config.get("git_executable")

    def execute_git_command(self, *args):
        """
        Git 명령어를 실행하는 메서드

        Args:
            *args: Git 명령어와 인자 목록
        """
        command = [self.git_executable] + list(args)
        command_str = " ".join(command).replace("\\", "/")
        log_manager.logger.info(f"명령어 실행: {command_str}")

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            log_manager.logger.info(f"명령어 출력: {result.stdout.strip()}")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            log_manager.logger.error(f"명령어 실패: {e.stderr.strip()}")
            raise GitSyncError(f"Git 명령어 실패: {e.stderr.strip()}")

    def sync_repository(self):
        """
        로컬 레포지토리를 원격 GitHub 레포지토리와 동기화하는 메서드
        """
        try:
            log_manager.logger.hr("레포지토리 동기화 시작", 1)

            # Git 초기화 (초기화가 필요한 경우)
            if not os.path.exists(".git"):
                log_manager.logger.info("Git 레포지토리 초기화")
                self.execute_git_command("init")
                self.execute_git_command("remote", "add", "origin", self.repository)

            # 최신 코드 가져오기 및 동기화
            log_manager.logger.info("최신 코드 가져오기")
            self.execute_git_command("fetch", "origin", self.branch)
            self.execute_git_command("reset", "--hard", f"origin/{self.branch}")

            log_manager.logger.hr("레포지토리 동기화 완료", 1)
        except GitSyncError as e:
            log_manager.logger.error(f"Git 동기화 중 오류 발생: {e}")
            raise

class GitSyncStrategy:
    """
    Git 레포지토리 동기화 전략 클래스
    """
    def sync(self, manager: GitManager):
        raise NotImplementedError

class PullStrategy(GitSyncStrategy):
    """
    기본 동기화 전략: Pull을 이용한 동기화
    """
    def sync(self, manager: GitManager):
        manager.sync_repository()

# 실행 예시
if __name__ == "__main__":
    git_manager = GitManager()
    pull_strategy = PullStrategy()

    try:
        pull_strategy.sync(git_manager)
        print("레포지토리 동기화 성공!")
    except Exception as e:
        log_manager.logger.error(f"레포지토리 동기화 실패: {e}")
