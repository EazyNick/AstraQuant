# 🚀 AstraQuant - AI 주식 자동매매 시스템

**Transformer 기반 PPO 강화학습을 활용한 지능형 주식 트레이딩 시스템**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [시스템 아키텍처](#-시스템-아키텍처)
- [설치 및 설정](#-설치-및-설정)
- [사용 방법](#-사용-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [설정 파일](#-설정-파일)
- [성능 및 결과](#-성능-및-결과)
- [기술 세부사항](#-기술-세부사항)
- [문제 해결](#-문제-해결)
- [기여 방법](#-기여-방법)
- [라이선스](#-라이선스)

---

## 🎯 프로젝트 개요

AstraQuant는 **PPO(Proximal Policy Optimization) 강화학습**과 **Transformer 신경망**을 결합하여 구현된 AI 주식 자동매매 시스템입니다. 시장 데이터를 분석하여 최적의 매매 타이밍을 학습하고, 실시간으로 거래 결정을 내릴 수 있는 지능형 트레이딩 에이전트를 제공합니다.

### 🎨 핵심 철학

- **단순성**: 복잡한 매매 전략을 3가지 명확한 액션으로 단순화
- **안정성**: PPO 알고리즘을 통한 안정적인 학습 과정
- **확장성**: 모듈화된 구조로 새로운 전략 및 지표 추가 용이
- **투명성**: 상세한 로깅과 TensorBoard 시각화 제공

---

## ✨ 주요 특징

### 🧠 AI 기술

- **PPO 강화학습**: 안정적이고 효율적인 정책 학습
- **Transformer 아키텍처**: 시계열 데이터의 장기 의존성 포착
- **Actor-Critic 구조**: 정책과 가치 함수의 균형잡힌 학습
- **Epsilon-Greedy 탐험**: 탐험과 활용의 적절한 균형

### 💹 트레이딩 시스템

- **3가지 액션**: 관망(Hold), 전부매수(Buy All), 전부매도(Sell All)
- **복합 보상 함수**: 단기/장기 수익률, 미래 가격 변화, 거래 수수료 등 종합 고려
- **리스크 관리**: 포트폴리오 밸런싱 및 손실 제한
- **실시간 분석**: 다양한 기술적 지표 활용

### 🔧 기술적 특징

- **모듈화 설계**: 각 컴포넌트의 독립적 개발 및 테스트 가능
- **GPU/CPU 자동 선택**: 하드웨어 환경에 맞는 자동 최적화
- **체크포인트 시스템**: 학습 중단 시 이어서 진행 가능
- **메모리 최적화**: 효율적인 메모리 사용량 관리

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      AstraQuant Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Data       │    │  Environment│    │  Agent      │          │
│  │  Loader     │───▶│  (Gym)      │◀──▶│  (PPO)      │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                             │                  │                │
│  ┌─────────────┐            │         ┌─────────────┐          │
│  │  Technical  │            │         │  Actor      │          │
│  │  Indicators │            │         │ (Transformer)│          │
│  └─────────────┘            │         └─────────────┘          │
│                             │                  │                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Portfolio  │    │  Reward     │    │  Critic     │          │
│  │  Manager    │◀───│  System     │    │ (Transformer)│          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │  Logger     │    │  TensorBoard│    │  Visualizer │          │
│  │  System     │    │  Metrics    │    │  & Analyzer │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 설치 및 설정

### 시스템 요구사항

- **Python**: 3.10 이상
- **메모리**: 8GB 이상 권장
- **GPU**: CUDA 지원 GPU (선택사항, CPU도 지원)
- **저장공간**: 최소 2GB 이상

### 설치 과정

1. **저장소 복제**

   ```bash
   git clone https://github.com/EazyNick/AstraQuant.git
   cd AstraQuant
   ```

2. **가상환경 생성 및 활성화**

   ```bash
   conda create -n AstraQuant python=3.10
   conda activate AstraQuant
   ```

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

### 주요 의존성 패키지

- **torch**: 2.6.0+ (딥러닝 프레임워크)
- **gym**: 0.26.2+ (강화학습 환경)
- **pandas**: 2.2.3+ (데이터 처리)
- **numpy**: 2.2.2+ (수치 계산)
- **tensorboard**: 로깅 및 시각화
- **yfinance**: 주식 데이터 수집
- **pandas_ta**: 기술적 지표 계산

---

## 🎮 사용 방법

### 1. 데이터 준비

주식 데이터를 다운로드하고 전처리합니다:

```bash
# 주피터 노트북 실행
jupyter notebook data/data_cursor.ipynb
```

또는 Python 스크립트로:

```python
# 데이터 로더 테스트
python data/data_loader.py
```

### 2. 모델 훈련

**기본 훈련:**

```bash
python main_train.py
```

**설정 사용자 정의:**

```bash
# config/config.yaml 파일 수정 후 실행
python main_train.py
```

**훈련 모니터링:**

```bash
# 별도 터미널에서 실행
tensorboard --logdir=logs/training
```

### 3. 모델 예측

**기본 예측:**

```bash
python main_predict.py
```

**사용자 정의 예측:**

```bash
python main_predict.py --model_path output/ppo_stock_trader_episode_1000.pth --test_data data/cursor_csv/AMZN_test_data.csv
```

### 4. 간단한 예측 실행

```bash
python simple_predict.py
```

---

## 📁 프로젝트 구조

```
AstraQuant/
├── 📁 agents/                    # 강화학습 에이전트
│   ├── __init__.py
│   └── actor_critic_agent.py     # PPO 에이전트 구현
├── 📁 config/                    # 설정 파일
│   ├── __init__.py
│   ├── config.py                 # 설정 매니저
│   └── config.yaml              # 주요 설정 파일
├── 📁 data/                      # 데이터 관련
│   ├── __init__.py
│   ├── data_loader.py           # 데이터 로더
│   ├── csv/                     # 원본 CSV 데이터
│   ├── cursor_csv/              # 전처리된 데이터
│   └── *.ipynb                  # 데이터 분석 노트북
├── 📁 env/                       # 거래 환경
│   ├── __init__.py
│   └── stock_env.py             # 주식 거래 환경 (Gym)
├── 📁 models/                    # 신경망 모델
│   ├── __init__.py
│   ├── actor_network.py         # Actor 네트워크
│   ├── critic_network.py        # Critic 네트워크
│   └── positionalencoding.py    # 포지셔널 인코딩
├── 📁 predict/                   # 예측 관련
│   ├── __init__.py
│   ├── model_loader.py          # 모델 로더
│   ├── predictor.py             # 예측기
│   ├── portfolio_analyzer.py    # 포트폴리오 분석
│   ├── visualizer.py            # 시각화
│   └── result_saver.py          # 결과 저장
├── 📁 training/                  # 훈련 관련
│   ├── __init__.py
│   └── train.py                 # 훈련 매니저
├── 📁 logs/                      # 로깅 시스템
│   ├── __init__.py
│   ├── logger.py                # 로거 설정
│   └── 📁 log/                  # 로그 파일들
├── 📁 output/                    # 결과 출력
│   └── *.pth                    # 훈련된 모델들
├── 📁 manage/                    # 관리 유틸리티
│   ├── __init__.py
│   └── directory.py             # 경로 관리
├── 📁 utils/                     # 유틸리티 함수
│   └── __init__.py
├── main_train.py                # 메인 훈련 스크립트
├── main_predict.py              # 메인 예측 스크립트
├── simple_predict.py            # 간단한 예측 스크립트
├── requirements.txt             # 의존성 패키지
└── README.md                    # 프로젝트 문서
```

## 📊 성능 및 결과

### 훈련 메트릭

- **포트폴리오 가치 변화**: 시간에 따른 포트폴리오 가치 추이
- **보상 함수**: 단기/장기 수익률, 거래 수수료 등 종합 평가
- **액션 분포**: 각 액션(관망/매수/매도)의 선택 빈도
- **손실 함수**: Actor/Critic 손실 값 추이

### 예측 결과

예측 결과는 다음과 같이 저장됩니다:

- **CSV 파일**: `output/model_predictions.csv`
- **로그 파일**: `logs/log/` 디렉토리
- **시각화**: 자동 생성되는 차트 및 그래프

### 성과 분석

- **Buy & Hold 전략과 비교**: 단순 매수 후 보유 전략과 성과 비교
- **샤프 비율**: 위험 대비 수익률 평가
- **최대 손실**: 최대 낙폭 분석
- **승률**: 수익 거래 비율

---

## 🔬 기술 세부사항

### 강화학습 알고리즘

**PPO (Proximal Policy Optimization)**

- 정책 기울기 방법의 안정성 개선
- 클리핑을 통한 보수적인 정책 업데이트
- Actor-Critic 구조로 효율적인 학습

### 신경망 아키텍처

**Transformer 기반 설계**

- Self-Attention 메커니즘으로 시계열 패턴 학습
- Positional Encoding으로 시간 정보 보존
- Multi-Head Attention으로 다양한 관점 분석

### 보상 함수

**복합 보상 시스템**

1. **단기 수익률 보상**: 포트폴리오 가치 변화율
2. **장기 수익률 보상**: 초기 대비 현재 수익률
3. **보유 주식 보상**: 가격 변화에 따른 보상
4. **미래 가격 보상**: 매매 후 가격 변화 예측
5. **거래 수수료 패널티**: 과도한 거래 억제
6. **수익 실현 보상**: 수익 구간에서 매도 시 보상
7. **손실 절단 보상**: 손실 구간에서 매도 시 보상

### 데이터 처리

**전처리 과정**

- 정규화 및 스케일링
- 결측치 및 이상치 처리
- 기술적 지표 계산
- 시계열 윈도우 생성

---

## 🔧 문제 해결

### 자주 발생하는 문제

**1. CUDA 메모리 부족**

```bash
# 배치 크기 줄이기
# config.yaml에서 batch_size: 8로 변경
```

**2. 모델 로딩 오류**

```bash
# 모델 파일 경로 확인
ls output/
python main_predict.py --model_path output/your_model.pth
```

**3. 데이터 파일 없음**

```bash
# 데이터 디렉토리 생성
mkdir -p data/cursor_csv
# 데이터 다운로드 실행
jupyter notebook data/data_cursor.ipynb
```

### 성능 최적화

**GPU 사용 최적화**

- 적절한 배치 크기 설정
- 메모리 사용량 모니터링
- 정기적인 메모리 정리

**CPU 사용 최적화**

- 멀티프로세싱 활용
- 메모리 효율적인 데이터 로딩
- 캐싱 전략 적용

---

## 🤝 기여 방법

### 기여 가이드라인

1. **이슈 생성**: 버그 리포트 또는 기능 제안
2. **포크 & 브랜치**: 개발용 브랜치 생성
3. **코드 작성**: 기존 코드 스타일 준수
4. **테스트**: 충분한 테스트 수행
5. **Pull Request**: 상세한 설명과 함께 PR 생성

## 📈 향후 개발 계획

### 단기 목표

- [ ] 다양한 기술적 지표 추가
- [ ] 백테스팅 시스템 구현
- [ ] 실시간 데이터 연동
- [ ] 웹 대시보드 개발

### 장기 목표

- [ ] 다중 자산 포트폴리오 지원
- [ ] 옵션 및 파생상품 거래
- [ ] 자동화된 하이퍼파라미터 튜닝
- [ ] 클라우드 배포 및 스케일링

---

## 📞 지원 및 문의

### 문의 채널

- **이메일**: [kkkygsos@naver.com](mailto:kkkygsos@naver.com)
- **GitHub Issues**: [프로젝트 이슈 페이지](https://github.com/EazyNick/AstraQuant/issues)
- **GitHub Discussions**: 질문 및 토론

---

## 📄 라이선스

이 프로젝트는 **Apache License 2.0** 하에 배포됩니다.

```
Copyright 2024 EazyNick

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 🏆 인정 사항

### 참조 논문

- Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017)
- Vaswani, A., et al. "Attention is All You Need." NIPS 2017

### 오픈소스 기여

- PyTorch 팀의 딥러닝 프레임워크
- OpenAI Gym의 강화학습 환경
- Hugging Face의 Transformers 라이브러리

---

## 🌟 즐겨찾기

이 프로젝트가 도움이 되었다면 ⭐을 눌러주세요!

[![GitHub stars](https://img.shields.io/github/stars/EazyNick/AstraQuant.svg?style=social&label=Star)](https://github.com/EazyNick/AstraQuant)
[![GitHub forks](https://img.shields.io/github/forks/EazyNick/AstraQuant.svg?style=social&label=Fork)](https://github.com/EazyNick/AstraQuant)

---

_최종 업데이트: 2025년 07월_
