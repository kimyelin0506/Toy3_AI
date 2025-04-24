import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# (1) 감성 예측 결과 (sentiment_label 기준) 분포
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
# 4. 시각화 (예측 결과 vs 별점 기반 결과를 같은 위치에 나란히)

# 먼저 긍정/부정 레이블 통일
label_map = {0: 'Negative', 1: 'Positive'}

predicted_counts = df['sentiment_binary'].map(label_map).value_counts().reindex(['Negative', 'Positive'])
true_counts = df['true_sentiment'].map(label_map).value_counts().reindex(['Negative', 'Positive'])

# 데이터프레임으로 합치기
compare_df = pd.DataFrame({
    'Review Analysis': predicted_counts,
    'True (Rating)': true_counts
})

# plot
compare_df.plot(kind='bar', figsize=(8, 5), color=['skyblue', 'lightgreen'])
plt.title('Predicted vs True Sentiment (Based on Rating)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.legend(title='Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --------------------------------------------
# (추가) 비율(%)로 표현한 버전

# 총 개수로 나눠서 비율 계산
predicted_ratio = predicted_counts / predicted_counts.sum() * 100
true_ratio = true_counts / true_counts.sum() * 100

# 데이터프레임으로 합치기
compare_ratio_df = pd.DataFrame({
    'Review Analysis (%)': predicted_ratio,
    'True (Rating) (%)': true_ratio
})

# plot
compare_ratio_df.plot(kind='bar', figsize=(8, 5), color=['dodgerblue', 'mediumseagreen'])
plt.title('Review Analysis vs True Sentiment (Based on Rating) - Percentage')
plt.xlabel('Sentiment')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 100)  # y축 0~100%로 고정
plt.show()


# (3) 감성 점수 분포 (히스토그램)
plt.figure(figsize=(6, 4))
sns.histplot(df[df['sentiment_binary'] != -1]['sentiment_score'], kde=True, bins=30, color="skyblue")
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# (4) rating과 sentiment_score 차이 계산 후 박스플롯
df['rating_diff'] = df['rating'] - df['sentiment_score']

plt.figure(figsize=(8, 5))
sns.boxplot(x='sentiment_label', y='rating_diff', data=df)
plt.title('Rating Difference by Sentiment')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Difference (Rating - Sentiment Score)')
plt.show()

# # --------------------------------------------
# # 5. 예측 오류 데이터 추출
#
# # 긍정으로 예측했는데 실제는 부정 (예측 1, 실제 0)
# false_positive = df[(df['sentiment_binary'] == 1) & (df['true_sentiment'] == 0)]
#
# # 부정으로 예측했는데 실제는 긍정 (예측 0, 실제 1)
# false_negative = df[(df['sentiment_binary'] == 0) & (df['true_sentiment'] == 1)]
#
# # 둘 합치기
# error_cases = pd.concat([false_positive, false_negative])
#
# # 결과 확인 (선택사항)
# print(f"총 오류 케이스 수: {len(error_cases)}")  #총 오류 케이스 수: 27965
#
# # CSV 저장
# error_cases.to_csv('../../renttherunway_mismatch_cases.csv', index=False)
#
# print("오류 케이스 CSV 저장 완료!")
