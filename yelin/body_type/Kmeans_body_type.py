# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

"""
여성 신체 데이터(Bust, Waist, Hips, Height 등)를 이용해서:
**각 사람별 "개인 체형"**을 분류하고
K-means 클러스터링으로 비슷한 사람들을 그룹으로 묶고
각 그룹마다 대표 체형을 정해서,
"나의 체형"과 "군집 대표 체형" 둘 다 보여주는 작업

"나는 hourglass형인데, 내 그룹 대표는 straight & narrow야!" 이런 걸 알 수 있음
개인/군집을 비교해서 "나는 군집 대표랑 비슷한가?", "나는 특이한가?" 분석할 수도 있음
"""

"""
1	기본 전처리하고, Cup Size, Gender 인코딩
2	classify_body_type()로 개별 사람 체형 구하기
3	K-means로 군집화
4	군집마다 가장 흔한 body type 찾아서 대표로 지정
5	각 사람에게 '나의 체형'과 '군집 대표 체형' 둘 다 보여줌
"""

# 2. 데이터 불러오기 및 기본 전처리
data = pd.read_csv('women_body_with_bmi.csv')
data.columns = [col.strip() for col in data.columns]
data = data.dropna()

# Label Encoding
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

le_cup = LabelEncoder()
data['Cup Size'] = le_cup.fit_transform(data['Cup Size'])
# print(len(data))  # 20000
# # 12. BMI 계산해서 추가
# data['BMI'] = data['Weight'] / ( (data['Height'] / 100) ** 2 )
#
# # 13. CSV 파일로 저장
# data.to_csv('women_body_with_bmi.csv', index=False)

# 3. body_type 분류 함수
def classify_body_type(row):
    waist = row['Waist']
    hips = row['Hips']
    bust = row['Bust/Chest']
    height = row['Height']

    if bust > hips and waist / bust > 0.8:
        return 'body_type_apple'
    elif bust > hips and waist / bust <= 0.8:
        return 'body_type_full bust'
    elif hips > bust and waist / hips <= 0.75:
        return 'body_type_pear'
    elif abs(bust - hips) < 3 and waist / hips < 0.8:
        return 'body_type_hourglass'
    elif height < 157:  # 157cm 이하
        return 'body_type_petite'
    elif abs(bust - hips) < 2 and waist / hips >= 0.8:
        return 'body_type_straight & narrow'
    else:
        return 'body_type_athletic'

# # 4. 스케일링
# X = data.copy()
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 5. 군집화 (k-means 사용)
# n_clusters = 7  # 너가 원하는 군집 개수로 수정 가능
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# clusters = kmeans.fit_predict(X_scaled)
#
# data['cluster'] = clusters  # 군집 결과 붙이기
#
# # 6. 각 사람별로 body_type 계산
# data['individual_body_type'] = data.apply(classify_body_type, axis=1)
#
# # 7. 군집별 대표 body_type 결정
# cluster_to_bodytype = {}
#
# for cluster_id in sorted(data['cluster'].unique()):
#     subset = data[data['cluster'] == cluster_id]
#     most_common_type = subset['individual_body_type'].mode()[0]  # 가장 많은 타입
#     cluster_to_bodytype[cluster_id] = most_common_type
#
# # 8. 최종 결과 보기
# data['representative_body_type'] = data['cluster'].map(cluster_to_bodytype)
#
# print(data[['Bust/Chest', 'Waist', 'Hips', 'Height', 'cluster', 'individual_body_type', 'representative_body_type']].head())
#
# from sklearn.metrics import silhouette_score
# sil_score = silhouette_score(X_scaled, clusters)
# print(f"Silhouette Score: {sil_score:.4f}")
#
# # 6. 각 사람별로 body_type 계산
# data['individual_body_type'] = data.apply(classify_body_type, axis=1)
#
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # PCA로 2차원 축소
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# # 군집 결과 시각화
# plt.figure(figsize=(10,8))
# sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2', s=60, edgecolor='k')
# plt.title(f'K-Means Clustering (k={n_clusters})')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.show()
# 4. 사용할 feature만 선택 (허리, 힙, 가슴, 키)
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6290
# X = data.copy()  # Silhouette Score: 0.1577
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6624
# X = data[['Waist', 'Hips', 'Bust/Chest', 'Height']].copy()  # Silhouette Score: 0.2114
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6839
# X = data[['Waist', 'Hips', 'Bust/Chest']].copy()  # Silhouette Score: 0.3129
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6867
# X = data[['Waist', 'Hips', 'Bust/Chest', 'BMI']].copy()  # Silhouette Score: 0.2724
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6735
X = data[['BMI', 'Hips', 'Waist']].copy()  # Silhouette Score: 0.2911
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6867
# X = data[['BMI', 'Hips', 'Waist', 'Bust/Chest']].copy()  # Silhouette Score: 0.2724
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6962
# X = data[['BMI', 'Waist', 'Bust/Chest']].copy()  # Silhouette Score: 0.2934
# 📊 전체 데이터 기준 일치율 (Accuracy): 0.6444
# X = data[['BMI', 'Hips', 'Bust/Chest']].copy()  # Silhouette Score: 0.2850

# 5. 스케일링
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
X_scaled = X
# 6. 군집화 (k-means 사용)
n_clusters = 7  # 원하는 군집 개수로 수정 가능
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data['cluster'] = clusters  # 군집 결과 붙이기

# 7. 각 사람별로 body_type 계산
data['individual_body_type'] = data.apply(classify_body_type, axis=1)

# 8. 군집별 대표 body_type 결정
cluster_to_bodytype = {}

for cluster_id in sorted(data['cluster'].unique()):
    subset = data[data['cluster'] == cluster_id]
    most_common_type = subset['individual_body_type'].mode()[0]
    cluster_to_bodytype[cluster_id] = most_common_type

# 9. 최종 결과
data['representative_body_type'] = data['cluster'].map(cluster_to_bodytype)

print(data[['Bust/Chest', 'Waist', 'Hips', 'Height', 'cluster', 'individual_body_type', 'representative_body_type']].head())

# 10. 실루엣 점수
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {sil_score:.4f}")

# 11. PCA 시각화
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2', s=60, edgecolor='k')
plt.title(f'K-Means Clustering (k={n_clusters})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# 12. 상관 계수 계산 & 시각화
import seaborn as sns
import matplotlib.pyplot as plt

# 상관관계 계산
corr = data[['Waist', 'Hips', 'Bust/Chest', 'Height', 'BMI']].corr()

# 수치 출력
print("🔎 특성 간 상관 계수 테이블:")
print(corr)
"""
🔎 특성 간 상관 계수 테이블:
               Waist      Hips  Bust/Chest    Height       BMI
Waist       1.000000  0.126688   -0.011005  0.027538  0.045390
Hips        0.126688  1.000000   -0.005867 -0.049831  0.037500
Bust/Chest -0.011005 -0.005867    1.000000  0.020946 -0.042107
Height      0.027538 -0.049831    0.020946  1.000000 -0.515322
BMI         0.045390  0.037500   -0.042107 -0.515322  1.000000

Waist ↔ Hips 약한 양의 상관 (0.13)
Waist, Hips ↔ Bust/Chest 상관 거의 없음 (거의 0)
Height ↔ BMI 강한 음의 상관 ( -0.515 ) → 키가 작을수록 BMI가 높게 나올 확률 있음
BMI ↔ 나머지 전부 약하거나 거의 무관
"""
# 히트맵 시각화
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# 13. 군집별 개인 체형과 대표 체형 일치율 계산

# 개인과 군집 대표가 일치하는지 여부를 계산
data['match'] = data['individual_body_type'] == data['representative_body_type']

# 전체 일치율
overall_accuracy = data['match'].mean()
print(f"\n📊 전체 데이터 기준 일치율 (Accuracy): {overall_accuracy:.4f}")

# 군집별 일치율
cluster_match_rate = (
    data.groupby('cluster')['match']
    .mean()
    .reset_index()
    .rename(columns={'match': 'match_rate'})
)

# 군집별 대표 체형을 디코딩하여 매핑
cluster_to_bodytype_str = {cluster_id: body_type for cluster_id, body_type in cluster_to_bodytype.items()}

# 군집 번호를 대표 체형으로 변환 (디코딩)
cluster_match_rate['representative_body_type'] = cluster_match_rate['cluster'].map(cluster_to_bodytype_str)

# 군집 크기 계산
cluster_size = data['cluster'].value_counts().reset_index()
cluster_size.columns = ['cluster', 'cluster_size']

# 군집 크기와 일치율 결합 전에 'cluster' 컬럼을 int로 변환
cluster_match_rate['cluster'] = cluster_match_rate['cluster'].astype(int)
cluster_size['cluster'] = cluster_size['cluster'].astype(int)

# 군집 크기와 일치율 결합
cluster_match_rate = pd.merge(cluster_match_rate, cluster_size, on='cluster')

# 전체 군집 수에 대한 비율 계산
cluster_match_rate['cluster_percentage'] = (cluster_match_rate['cluster_size'] / len(data)) * 100

# 결과 출력
print("\n📊 군집별 대표 체형과 개인 체형 일치율 (군집 크기 포함):")
print(cluster_match_rate[['representative_body_type', 'match_rate', 'cluster_size', 'cluster_percentage']])

# 보기 쉽게 시각화
plt.figure(figsize=(8,6))
sns.barplot(x='cluster', y='match_rate', data=cluster_match_rate, hue='cluster', palette='viridis', legend=False)
plt.ylim(0,1)
plt.title('Cluster-wise Body Type Match Rate')
plt.xlabel('Cluster')
plt.ylabel('Match Rate')
plt.grid(True, axis='y')
plt.show()

"""
📊 군집별 대표 체형과 개인 체형 일치율:
   cluster  match_rate
0        0    1.000000
1        1    0.772663
2        2    0.595269
3        3    0.617907
4        4    0.331688
5        5    0.514148
6        6    0.902190
"""
# # 바디 타입을 문자열로 디코딩
# data['individual_body_type'] = le_cup.inverse_transform(data['individual_body_type'])
# data['representative_body_type'] = le_cup.inverse_transform(data['representative_body_type'])
#
# # 결과 출력
# print(data[['Bust/Chest', 'Waist', 'Hips', 'Height', 'cluster', 'individual_body_type', 'representative_body_type']].head())
