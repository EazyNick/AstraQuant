# 📈 주식 자동매매 시스템 (PPO + Transformer) (250407 최신화)

## 📌 프로젝트 개요
본 프로젝트는 **PPO(Proximal Policy Optimization) 강화학습 알고리즘**과 **트랜스포머(Transformer) 기반 신경망**을 활용하여 **주식 자동매매 시스템**을 구현하는 것을 목표로 합니다.

강화학습을 활용하여 **최적의 매매 전략**을 학습하고, 이를 실시간 시장 데이터에 적용하여 **매도(Sell) / 관망(Hold) / 매수(Buy)** 결정을 수행하는 것이 핵심 기능입니다.

---

## 🚀 모델 개요
### **1️⃣ PPO (Proximal Policy Optimization)**
본 프로젝트에서는 **PPO 알고리즘을 사용하여 최적의 거래 전략을 학습**합니다. PPO는 정책 기반(Policy Gradient)과 가치 기반(Value Function)을 결합한 Actor-Critic 구조를 사용하여 학습을 진행합니다.

- Actor (정책 네트워크): 주어진 상태에서 최적의 매매 결정을 수행하는 역할
- Critic (가치 네트워크): 특정 상태의 가치를 평가하여 Actor의 업데이트를 돕는 역할
- Clipped Objective Function을 사용하여 정책 업데이트가 과하게 변하지 않도록 안정성을 확보
- Advantage Estimation을 통해 학습 효율성을 향상

### **2️⃣ Transformer 기반 신경망**
기존 강화학습 모델에서는 CNN 또는 LSTM을 사용하지만, 본 프로젝트에서는 **트랜스포머 신경망을 활용하여 시계열 데이터를 처리**합니다.

- **자기회귀(Self-Attention) 메커니즘을 활용하여 장기적인 패턴을 학습**
- **CNN, LSTM보다 더 효과적으로 금융 데이터를 처리**
- **강화학습 환경에서 시퀀스 기반 매매 패턴을 학습하는데 적합**

---

## 🔧 실행 방법

### **1️⃣ 필수 라이브러리 설치**

프로젝트 실행 전에 필수 라이브러리를 설치해야 합니다.

```bash
pip install -r requirements.txt
```

---
✅ Python 버전: 본 프로젝트는 Python 3.10.16 환경에서 실행됩니다.

### **2️⃣ 데이터 다운로드**

프로젝트를 실행하기 전에 **주식 데이터를 다운로드하여 전처리**해야 합니다.

```bash
python data/data_s&p.ipynb
```

이 스크립트는 S&P 500 및 개별 주식(AAPL, TSLA 등)의 데이터를 다운로드하고, **학습 데이터(train) 및 테스트 데이터(test)** 로 저장합니다.

---

### **3️⃣ 모델 학습**

강화학습을 수행하여 **PPO 에이전트를 학습**합니다.

```bash
python training/train.py
```

훈련이 완료되면 `output/ppo_stock_trader.pth` 파일이 생성됩니다.

---

### **4️⃣ 모델 예측**

훈련된 모델을 사용하여 새로운 데이터를 기반으로 **매매 결정을 수행**합니다.

```bash
python main_predict.py --model_path output/ppo_stock_trader_episode_350.pth --test_data data/csv/005930.KS_combined_train_data.csv
```

실행 후, **날짜별 매매 신호(매수/매도/관망)와 확률 값**이 출력됩니다.

---

## 📊 평가 및 성능 비교

### **1️⃣ Buy & Hold 전략과 비교**

- **Buy & Hold 전략**: 초기 투자금을 한 번에 매수 후 보유하는 전략
- **PPO 강화학습 모델**: 강화학습을 통해 최적의 매매 타이밍을 학습한 전략

학습된 모델이 **Buy & Hold보다 높은 수익률을 달성하는지 비교**하여 성능을 평가합니다.

### **2️⃣ 모델 예측 결과 저장**

모델 예측 결과는 CSV 파일로 저장됩니다.

```bash
output/model_predictions.csv
```

해당 파일을 활용하여 **매매 전략 성과 분석 및 시각화**가 가능합니다.

---

## 📌 결론

본 프로젝트는 강화학습을 활용하여 **최적의 주식 매매 전략을 학습하고 적용하는 시스템**을 구축합니다.

- **PPO 알고리즘을 적용한 강화학습 모델**
- **Transformer 기반 신경망을 활용하여 시계열 데이터 처리**
- **Buy & Hold 전략과 비교하여 성능 평가**
- **주식 시장 데이터(S&P 500, AAPL, TSLA 등)를 활용한 실전 적용 가능**

앞으로 추가적인 개선을 통해 **강화학습 기반 자동매매의 효율성을 극대화**할 수 있도록 발전시킬 계획입니다. 🚀

## 📌 업데이트 예정

- 기타 최적화


## 📌 문의 및 이슈 등록

- 문의: [kkkygsos@naver.com](mailto:kkkygsos@naver.com)
- 이슈 등록: [GitHub Issues](https://github.com/EazyNick/AstraQuant/issues)

## 📌 라이선스

본 프로젝트는 **Apache License 2.0**을 따릅니다.

자세한 내용은 [LICENSE](http://www.apache.org/licenses/LICENSE-2.0)에서 확인할 수 있습니다.

