import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
별점과 감성 예측이 일치한 비율 (정확도): 0.8547
"""
# 데이터 가져오기
df = pd.read_csv('../../renttherunway_data_with_sentiment.csv')
df = df.dropna(subset=['review_text'])  # NaN 값 제거

# --------------------------------------------
# 1. 별점 기준 감정 레이블 생성 (true_sentiment 컬럼 업데이트)
def true_sentiment(rating):
    if rating >= 8:
        return 1  # 긍정
    else:
        return 0  # 부정

df['true_sentiment'] = df['rating'].apply(true_sentiment)

# 2. sentiment_binary와 true_sentiment 비교하여 match 컬럼 생성
df['match'] = (df['sentiment_binary'] == df['true_sentiment']).astype(int)

# 3. 최종 정확도 출력
accuracy = df['match'].mean()
print(f"별점과 감성 예측이 일치한 비율 (정확도): {accuracy:.4f}")

# --------------------------------------------
# 4. 시각화

# (1) 긍정 vs 부정 막대 그래프
sentiment_counts = df['sentiment_label'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Blues_d", hue=sentiment_counts.index, legend=False)
plt.title('Sentiment Distribution (Positive vs Negative)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.show()

# (2) 감성 점수 분포 (히스토그램)
plt.figure(figsize=(6, 4))
sns.histplot(df[df['sentiment_binary'] != -1]['sentiment_score'], kde=True, bins=30, color="skyblue")
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# (3) rating과 sentiment_score 차이 계산 후 박스플롯
df['rating_diff'] = df['rating'] - df['sentiment_score']

plt.figure(figsize=(8, 5))
sns.boxplot(x='sentiment_label', y='rating_diff', data=df)
plt.title('Rating Difference by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Difference (Rating - Sentiment Score)')
plt.show()
