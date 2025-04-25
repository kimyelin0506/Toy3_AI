# 리뷰 감성 분석 및 별점 일치도 서비스
## 프로젝트 개요

렌트더런웨이(Rent the Runway)의 리뷰 데이터를 활용하여 **리뷰 내용(자연어)**과 **별점(정량적 평가)** 간의 일치 여부를 분석하는 프로젝트입니다.  
Hugging Face의 `DistilBERT` 모델을 이용해 감성 분류를 수행하고, 결과를 시각화 및 평가합니다.

---

## 사용 기술

- **NLP 모델**: DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- **프레임워크**: PyTorch (`transformers`, `pipeline`)
- **시각화**: `seaborn`, `matplotlib`
- **데이터 처리**: `pandas`

---

## DistilBERT 모델

### 1. DistilBERT란?

- **DistilBERT**는 BERT 모델을 경량화한 버전으로, 지식 증류(Knowledge Distillation) 기법을 사용해 BERT의 성능을 유지하면서도 속도와 크기를 최적화한 모델입니다.
- DistilBERT는 **BERT보다 40% 더 작고 60% 더 빠르며**, 성능 손실은 약 3% 미만입니다.

### 2. DistilBERT의 특징

- **경량화 모델**: 빠르고 가벼운 성능을 자랑하며, 모바일과 웹 환경에서 실시간으로 대량 데이터를 처리하는 데 적합합니다.
- **사전 학습 모델 활용**: 이미 학습된 감성 분류 모델을 활용하여 별도의 학습 없이 즉시 사용할 수 있습니다.
- **확률적 예측**: Softmax를 통해 감성 예측 결과를 확률로 변환하여 분류합니다.

### 3. DistilBERT의 구조

- **입력**: 텍스트 데이터는 토큰화 과정을 거쳐 벡터 형태로 변환됩니다.
- **인코딩**: 입력된 벡터는 여러 층의 Transformer 인코더를 통과하여 문맥을 이해합니다.
- **출력**: [CLS] 토큰 벡터를 사용하여 문장의 감성을 예측합니다.

---

## 프로젝트 구성

### `sentiment_analysis.py`
사전 학습된 감성 분석 모델을 통해 리뷰 데이터를 분석합니다.

#### 주요 기능

1. 리뷰 텍스트 전처리 및 안전 분석 함수 정의
2. 감성 분석 수행 (긍정 / 부정 라벨 및 확신도 score)
3. 리뷰 결과를 기존 CSV에 병합하여 저장 (`renttherunway_data_with_sentiment.csv`)
4. 별점(rating)을 기반으로 한 감정값 생성 (4점 이상 긍정)
5. 감성 분석 결과와 실제 별점 감정의 **일치 여부(정확도)** 계산
6. 감성 분포 및 점수 시각화

---

### `result_analysis.py`

분석 결과가 담긴 CSV 파일(`renttherunway_data_with_sentiment.csv`)을 바탕으로 감성 분석의 신뢰도를 평가하고 시각화합니다.

#### 주요 기능

1. **별점 기준 변경 (8점 이상 → 긍정)** 으로 `true_sentiment` 정의
2. `sentiment_binary`와 비교해 감성 예측 정확도 재계산
3. 감성 예측 결과 vs 실제 감정(별점) 비교 시각화 (막대그래프, 박스플롯 등)
4. 예측이 틀린 리뷰 추출 및 저장 (`renttherunway_mismatch_cases.csv`)

---

## 주요 시각화

- 감성 예측 분포 (`Positive`, `Negative`)
- 예측 vs 실제 감성 비교 (개수 및 퍼센트)
- 감성 점수 분포 (히스토그램)
- 별점과 감성 점수 차이 (Boxplot)

---

## 주요 성과

- 감성 예측 정확도 계산 가능
- 감성 라벨과 별점의 **불일치 리뷰 탐지**
- 실제 서비스에서 리뷰 품질을 자동으로 판단하는 데 활용 가능

---

## 실행 예시

```bash
# 감성 분석 수행
python sentiment_analysis.py

# 결과 평가 및 시각화
python result_analysis.py
