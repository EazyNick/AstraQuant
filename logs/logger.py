"""
LogManager Module

싱글톤 패턴으로 설계되어 로그 설정을 중앙에서 관리하며, 다음과 같은 주요 기능을 포함합니다:
- 로그 파일 생성 및 관리
- 컬러 로그 포맷 지원
- 오래된 로그 파일 자동 정리

사용 예시:
    log_manager = LogManager()
    log_manager.logger.info("This is an info message")
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import glob
import colorlog

class LogManager:
    """
    로그 관리 클래스

    싱글톤 패턴으로 설계되었으며, 컬러 로그 포맷 및 로그 파일 정리를 지원합니다.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LogManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self, directory=r'D:\Python\AstraQuant\logs\log', max_files=10):
        """
        LogManager 초기화

        Args:
            directory (str): 로그 파일을 저장할 디렉토리
            max_files (int): 유지할 최대 로그 파일 개수
        """
        if not hasattr(self, 'initialized'):  # 이 인스턴스가 초기화되었는지 확인
            self.directory = os.path.join(os.path.dirname(__file__), "log") or directory
            print(self.directory)
            self.max_files = max_files
            self._timestamp = self._init_timestamp()
            self.logger = self._init_logger()
            self._bind_hr_to_logger()
            self.clean_up_logs()
            self.initialized = True

    def _init_timestamp(self):
        """ 타임스탬프 초기화 """
        return datetime.now().strftime("%Y%m%d-%H%M%S")


    def _init_logger(self):
        """ 로거 초기화 """
        logger = logging.getLogger('AstraQuant')

        log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red', 
            'CRITICAL': 'red,bg_white'
        }

        formatter = colorlog.ColoredFormatter(
            '[%(log_color)s%(asctime)s.%(msecs)03d][%(levelname).1s][%(filename)s(%(funcName)s):%(lineno)d] %(message)s',
            log_colors=log_colors_config,
            datefmt='%Y-%m-%d %H:%M:%S',
            reset=True,
            secondary_log_colors={}
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        print(f'실행한 파일명: {os.path.basename(sys.argv[0])}')

        # 로그 저장 주소
        logfile = f"{self._timestamp}_{os.path.basename(sys.argv[0])}.log"
        logpath = Path(self.directory)

        if not logpath.exists():
            os.makedirs(logpath)

        file_handler = logging.FileHandler(logpath / logfile, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(file_handler)

        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        logger.debug('Logger initialized')

        return logger

    def hr(self, message="", level=1):
        """
        구분선을 출력하는 메서드.

        Args:
            message (str): 구분선 위에 출력할 메시지 (기본값: 빈 문자열)
            level (int): 구분선 길이 조정 (기본값: 1)
        """
        line = "=" * (level * 20)  # 구분선 길이 설정
        if message:
            self.logger.info(f"\n{line}\n{message}\n{line}")
        else:
            self.logger.info(f"\n{line}")

    def _bind_hr_to_logger(self):
        """
        hr 메서드를 logger 객체에 동적으로 추가합니다.
        """
        self.logger.hr = self.hr  # logger 객체에 hr 메서드 바인딩

    def get_timestamp(self) -> str:
        """ 시작 타임스탬프 반환 """
        return self._timestamp

    def clean_up_logs(self):
        """
        오래된 로그 파일 정리

        설정된 디렉토리에서 최대 파일 개수를 초과하는 오래된 로그 파일을 삭제합니다.
        """
        # 디렉토리 내의 특정 패턴의 파일 목록을 가져옵니다.
        files = glob.glob(os.path.join(self.directory, '*.log'))

        # 파일을 생성 시간에 따라 정렬합니다.
        files.sort(key=os.path.getmtime)

        self.logger.info('clean_up_logs')

        # 지정된 개수를 초과하는 파일이 있다면, 가장 오래된 파일부터 삭제합니다.
        while len(files) > self.max_files:
            os.remove(files.pop(0))  # 가장 오래된 파일을 삭제하고 목록에서 제거합니다.

if __name__ == "__main__":
    log_manager = LogManager()
    log_manager.logger.info("This is an info message for testing purposes.")
    log_manager.logger.debug("This is a debug message for testing purposes.")
    log_manager.logger.error("This is an error message for testing purposes.")
    log_manager.logger.hr("Section Start", level=2)

    print(f"Timestamp: {log_manager.get_timestamp()}")
