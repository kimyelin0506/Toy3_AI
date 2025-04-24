import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
"""
[DistilBERT 모델을 이용한 이진 감성 분류기]
렌트더런웨이(renttherunway) 데이터셋의 리뷰(review_text)에 대해 감성 분석
그 결과를 별점(rating)과 비교해 정확도를 측정하는 파이프라인

(1) 필요한 라이브러리 불러오기
- 데이터 다루기: numpy, pandas
- 딥러닝 처리: torch
- 시각화: seaborn, matplotlib
- 감성 분석 모델 불러오기: transformers.pipeline

(2) 사전 학습된 감성 분석 모델 로드
- DistilBERT 기반으로 학습된 영어 감성 분석 모델을 불러옴.
- framework="pt"로 지정해서 PyTorch를 사용.
- 이 sentiment_analyzer가 문장을 넣으면 긍정/부정 분류를 해줘.

(3) 데이터 불러오기 및 전처리
- renttherunway_data.csv를 읽어옴.
- 리뷰 텍스트가 없는 행(NaN)은 제거.

(4) 리뷰 텍스트 준비
- 혹시 review_text가 문자열이 아닐 수도 있어서, 안전하게 문자열만 남김.
- 텍스트만 따로 리스트로 추출하는 단계.

(5) 감성 분석 함수 정의
- 텍스트가 비었거나, 너무 길면(1000자 초과) 자르고,
- 모델 예측을 호출하고 에러를 처리하는 안전한 함수야.
- 결과가 {label: 긍정/부정, score: 확신도} 형태

(6) 텍스트별 감성 분석
- 각 리뷰 텍스트에 대해 감성 분석을 수행.
- 분석 결과
    - 라벨(label): 'POSITIVE' 또는 'NEGATIVE'
    - 점수(score): 확신 정도 (0~1 사이)
    - 이진(binary_labels): POSITIVE → 1, NEGATIVE → 0, 알 수 없음 → -1 형태로 리스트에 저장.

(7) 결과를 데이터프레임에 추가
- 감성 분석 결과를 원래 df에 새로운 컬럼으로 추가

(8) 별점(rating) 기반으로 진짜 감성 만들기
- 별점(rating) 4점 이상이면 긍정(1), 그렇지 않으면 부정(0)으로 간주.

(9) 감성 예측 정확도 계산
- 감성 분석 결과와 별점 감성이 일치하는지 비교해서
- **정확도(accuracy)**를 출력.

(10) 결과 저장
- 감성 분석 결과를 포함한 CSV 파일로 저장.

(11) 감성 분석 결과 시각화
- 긍정/부정 비율 막대그래프
- 감성 점수 분포
- 별점 차이(boxplot)
"""

"""
DistilBERT + Fine-tuning된 Sentiment Classification 모델
기본 알고리즘은 **BERT(Bidirectional Encoder Representations from Transformers)**

그런데 DistilBERT는 BERT를 "경량화"한 버전

<정리>
- Transformer 기반 모델
- DistilBERT: (BERT의 축소판)
- SST-2 데이터셋으로 Fine-tuning된 binary classifier
- 내부적으로는 [CLS] 토큰을 가지고 전체 문장을 요약하고,
- 그걸 바탕으로 Positive / Negative를 예측

<흐름>
입력 문장 → DistilBERT 인코딩 → [CLS] 임베딩 → Linear Layer → Softmax → Positive/Negative
- [CLS] 토큰: 전체 문장을 요약한 하나의 벡터
- Linear Layer: 출력 차원은 2개 (긍정/부정)
- Softmax: 0~1 사이의 확률로 변환
- 결과: 높은 쪽 클래스를 선택 (ex: Positive 확률 0.93 → POSITIVE 예측)
"""

# 영어용 사전학습 감성분석 모델 불러오기
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)

# 데이터 가져오기
df = pd.read_csv('../../renttherunway_data.csv')
df = df.dropna(subset=['review_text'])  # NaN 값 제거

# 데이터 10분의 1만 사용
# df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

# 앞에서 10개만 가져오기
# df = df.head(10)

# 데이터가 문자열 형식이 맞는지 확인하고, 텍스트에만 분석을 적용하도록 처리
texts = df['review_text'].apply(lambda x: x if isinstance(x, str) else "")

MAX_LENGTH = 512  # DistilBERT 최대 길이

def safe_sentiment_analyze(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {'label': 'UNKNOWN', 'score': 0.0}
    if len(text) > 1000:  # 글자 수가 너무 많으면 텍스트 자르기 (512 토큰 대략 1000자 이내)
        text = text[:1000]
    try:
        result = sentiment_analyzer(text)[0]
        return result
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return {'label': 'UNKNOWN', 'score': 0.0}


# 결과 저장용 리스트
labels = []
scores = []
binary_labels = []

# 후기 감성 분석
for text in texts:
    if text:  # 텍스트가 비어있지 않으면 감성 분석
        result = safe_sentiment_analyze(text)
        label = result['label']
        score = result['score']
        labels.append(label)
        scores.append(score)
        binary_labels.append(1 if label == 'POSITIVE' else 0 if label == 'NEGATIVE' else -1)
    else:
        # 비어있는 텍스트는 제외하거나 기본값 처리
        labels.append('UNKNOWN')
        scores.append(0.0)
        binary_labels.append(-1)

# 결과를 데이터프레임에 추가
df['sentiment_label'] = labels
df['sentiment_score'] = scores
df['sentiment_binary'] = binary_labels

# --------------------------------------------
# 별점과 감정 예측 비교 (정확도 측정)
#
# # 별점 기준 설정 (4, 5는 긍정, 1, 2, 3은 부정)
# def true_sentiment(rating):
#     if rating >= 4:
#         return 1  # 긍정
#     else:
#         return 0  # 부정
#
# # 별점으로 실제 감정 추정
# df['true_sentiment'] = df['rating'].apply(true_sentiment)
#
# # 예측과 실제 비교
# df['match'] = (df['sentiment_binary'] == df['true_sentiment']).astype(int)
#
# # 최종 정확도
# accuracy = df['match'].mean()
# print(f"별점과 감성 예측이 일치한 비율 (정확도): {accuracy:.4f}")
#
# # 최종 CSV로 저장
# df.to_csv('../renttherunway_data_with_sentiment.csv', index=False)
#
# print("CSV 파일 저장 완료!")
#
# # 긍정과 부정의 분포를 막대 그래프로 시각화
# sentiment_counts = df['sentiment_label'].value_counts()
#
# plt.figure(figsize=(6, 4))
# sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Blues_d",  hue=sentiment_counts.index,  legend=False)
# plt.title('Sentiment Distribution (Positive vs Negative)')
# plt.xlabel('Sentiment')
# plt.ylabel('Number of Reviews')
# plt.show()
#
# # 긍정과 부정의 감성 점수 분포를 시각화
# plt.figure(figsize=(6, 4))
# sns.histplot(df[df['sentiment_binary'] != -1]['sentiment_score'], kde=True, bins=30, color="skyblue")
# plt.title('Sentiment Score Distribution')
# plt.xlabel('Sentiment Score')
# plt.ylabel('Frequency')
# plt.show()
#
# # 평가 예시 - 부정/긍정 예측 점수와 실제 rating 점수 비교
# df['rating_diff'] = df['rating'] - df['sentiment_score']
# plt.figure(figsize=(8, 5))
# sns.boxplot(x='sentiment_label', y='rating_diff', data=df)
# plt.title('Rating Difference by Sentiment')
# plt.xlabel('Sentiment')
# plt.ylabel('Difference (Rating - Sentiment Score)')
# plt.show()
