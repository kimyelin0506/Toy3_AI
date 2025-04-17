import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# 1. 파일 읽기
df = pd.read_csv("../renttherunway_data.csv")

# 2. 사용할 특성들만 뽑기
data = df[['fit', 'user_id', 'item_id', 'weight', 'rating', 'body type', 'height', 'size', 'age']].copy()

# 3. 컬럼명 변경
data.rename(columns={
    'weight':'weight (kg)',
    'body type':'body_type',
    'height':'height (cm)'
}, inplace=True)

# 4. 단위 변환 함수 정의
def lb_to_kg_trans(lb_str):
    if isinstance(lb_str, str) and 'lbs' in lb_str:
        number = int(lb_str.replace('lbs', '').strip())
        return round(number * 0.453592, 2)
    return None

def height_trans(height):
    if pd.isna(height):
        return height
    if isinstance(height, str) and "'" in height and "\"" in height:
        feet = float(height.split("'")[0]) * 30.48
        inch = float(height.split("'")[1].replace("\"", "").strip()) * 2.54
        return feet + inch
    return None

def rating_scale_trans(rating):
    return rating / 2

# 5. 단위 변환 적용
data['weight (kg)'] = data['weight (kg)'].apply(lb_to_kg_trans)
data['height (cm)'] = data['height (cm)'].apply(height_trans)
data['rating'] = data['rating'].apply(rating_scale_trans)

# 'fit' 컬럼에서 'fit'이 아닌 값을 가진 행 삭제
# data = data[data['fit'] == 'fit']
# 매핑
fit_mapping = {
    'small' : 0,
    'fit' : 2,
    'large' : 1
}
data['fit'] = data.loc[:, 'fit'].map(fit_mapping)

# 6. 원핫 인코딩 (이 타이밍에 진행해야 이후 body_type 컬럼이 사라진 걸 인지 가능)
data = pd.get_dummies(data, columns=['body_type'])
# data = pd.get_dummies(data, columns=['fit', 'body_type'])

# 7. 결측치 처리
# - weight는 같은 size 평균으로 채우기
size_weight_means = data.groupby('size')['weight (kg)'].mean()
def fill_weight(row):
    if pd.isna(row['weight (kg)']):
        return size_weight_means.get(row['size'], None)
    else:
        return row['weight (kg)']
data['weight (kg)'] = data.apply(fill_weight, axis=1)

# - 나머지 수치형 평균값으로 채우기
# data['weight (kg)'] = data['weight (kg)'].fillna(data['weight (kg)'].mean())
# data['height (cm)'] = data['height (cm)'].fillna(data['height (cm)'].mean())
# data['rating'] = data['rating'].fillna(data['rating'].mean())
# data['age'] = data['age'].fillna(data['age'].mean())
data = data.dropna(subset=['weight (kg)', 'height (cm)', 'rating', 'age'])
print(data)
data = data[data['fit'] == 2]
data = data[data['rating'] == 5]

df = pd.read_csv('../renttherunway_data.csv')

# # 스케일 조정

"""
좋은 질문이야! 일반적으로는 스케일링을 하면 KMeans나 다른 거리 기반 알고리즘의 성능이 올라가는 경우가 많아, 하지만 지금처럼 스케일링을 했더니 오히려 Silhouette Score가 떨어지는 경우는 종종 발생할 수 있어. 왜 그런지 아래에 이유들을 정리해줄게.

✅ KMeans에서 스케일링이 중요한 이유
KMeans는 유클리디안 거리(Euclidean Distance) 기반 알고리즘이라서 각 feature의 값의 크기 차이가 클 경우, 큰 값을 가지는 feature에 클러스터링이 편향되기 쉬워. 그래서 보통 스케일링은 도움이 돼.

하지만!

🔍 너의 경우처럼 스케일링 후 성능이 떨어지는 이유들
1. 스케일링 자체가 feature 의미를 날려버렸을 수 있음
예를 들어 fit, size, rating, weight, height 등의 컬럼은 서로 의미적으로 다르고, 특히 fit이나 size는 본질적으로 범주형 또는 순서형 변수야.

👉 근데 이걸 StandardScaler로 스케일링하면 단순히 평균 0, 분산 1로 변환되기 때문에, 클러스터링에서 의미 있는 정보가 희석될 수 있어.

예:

fit=2 (정사이즈), fit=0 (작다)는 의미상 엄청 다르지만, 스케일 후엔 단지 정규분포 상의 수치일 뿐임.

2. 스케일링이 군집 간 경계를 무너뜨림
원래는 클러스터들이 특정 feature 기준으로 자연스럽게 구분되고 있었는데, 스케일링으로 feature 간 상대적인 거리 구성이 바뀌면서 군집 간 경계가 모호해지는 경우도 있어.

3. 스케일된 값으로는 사람이 해석하기 어렵다
스케일 전엔 weight나 size 같은 컬럼이 직관적으로 클러스터 차이를 보여줄 수 있었는데, 스케일 후엔 모두 추상적인 숫자라서 클러스터의 특성을 구분하기 어려움 → 이로 인해 silhouette score도 저하.

4. rating, fit 등의 범주형 값까지 스케일링해서 생긴 부작용
KMeans는 연속형 수치에 강한데, 범주형 정보가 스케일링되면 의미 없는 거리 계산이 이뤄질 수 있어.

예:

fit = 0, 1, 2는 숫자처럼 보여도 범주형인데, 스케일러는 이걸 연속형으로 처리함 → 문제가 생김.


"""
# data = data.drop(columns=['user_id', 'item_id', 'rating', 'body_type_apple', 'body_type_athletic',
#        'body_type_full bust', 'body_type_hourglass', 'body_type_pear',
#        'body_type_petite', 'body_type_straight & narrow'])
original_df = data.copy()
sc = MinMaxScaler()
scaled_array = sc.fit_transform(original_df)
scaled_df = pd.DataFrame(scaled_array, columns=original_df.columns)

# original_df = data.copy()
# sc = StandardScaler()
# scaled_array = sc.fit_transform(original_df)
# scaled_df = pd.DataFrame(scaled_array, columns=original_df.columns)

"""# 상관 관계 확인
def corr_analysis(corr_input):
    if 0.0 <= corr_input <= 0.1:
        print("거의 상관 없음")
    elif 0.1 < corr_input <= 0.3:
        print("약한 상관관계 (weak)")
    elif 0.3 < corr_input <= 0.7:
        print("중간 정도 상관관계 (moderate)")
    elif 0.7 < corr_input <= 1.0:
        print("강한 상관관계 (strong)")
    else:
        print("상관계수 값이 범위를 벗어났습니다.")
print(scaled_df.columns)

# 상관관계 분석 대상 컬럼 리스트
columns = [col for col in scaled_df.columns]

# 모든 쌍 조합을 생성해서 비교
from itertools import combinations
for col1, col2 in combinations(columns, 2):
    corr = scaled_df[col1].corr(scaled_df[col2])
    print(f"[{col1} / {col2}] 상관 계수: {corr:.4f}")
    corr_analysis(corr)
    print("-" * 60)
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화할 컬럼만 추림 (필요 시 user_id, item_id 같은 컬럼은 제거)
exclude_cols = ['user_id', 'item_id']
columns = [col for col in scaled_df.columns if col not in exclude_cols]

# 상관계수 행렬 생성
corr_matrix = scaled_df[columns].corr()

# 히트맵 시각화
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.7})
plt.title("📊 Feature Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""

print(data.columns)
# 샘플링
# sampled_data = resample(scaled_df, n_samples=50000, random_state=42)
# sampled_data = sampled_data.drop(columns=['user_id', 'item_id', 'rating'])
# sampled_data = sampled_data[['height (cm)','weight (kg)', 'size', 'item_id', 'rating', 'fit']]
# sampled_data = sampled_data[['height (cm)', 'weight (kg)', 'size']]
# print(sampled_data)

# 학습
k=20
km = KMeans(n_clusters=k, random_state=42, init='k-means++')
km.fit(scaled_df)
labels = km.labels_
score = silhouette_score(scaled_df, labels)
print("[",k,"] score: ", score)
print(km.n_clusters)

import seaborn as sns

# KMeans 클러스터링 결과를 sampled_data에 추가
scaled_df['cluster'] = labels

# 클러스터링된 결과 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(data=scaled_df, x='weight (kg)', y='height (cm)', hue='cluster', palette='viridis', s=100, alpha=0.7, edgecolor='black')
plt.title(f"KMeans Clustering Results (k={k})", fontsize=16)
plt.xlabel("Weight (kg)")
plt.ylabel("Size")
plt.legend(title='Cluster', loc='best')
plt.tight_layout()
plt.show()
# ========================================
# 📈 최적 k 찾기 (Elbow + Silhouette Score)
# ========================================

# inertias = []
# silhouette_scores = []
# K_range = range(2, 20)  # 일반적으로 2~10 사이에서 탐색
#
# for k in K_range:
#     km = KMeans(n_clusters=k, random_state=42)
#     km.fit(sampled_data)
#     inertias.append(km.inertia_)
#
#     labels = km.labels_
#     score = silhouette_score(sampled_data, labels)
#     silhouette_scores.append(score)
#     print("k: ",k,", score: ", score)
#
# # 📊 그래프 출력
# plt.figure(figsize=(14, 5))
#
# # 1. Elbow Method
# plt.subplot(1, 2, 1)
# plt.plot(K_range, inertias, 'o-', color='blue')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.grid(True)
#
# # 2. Silhouette Score
# plt.subplot(1, 2, 2)
# plt.plot(K_range, silhouette_scores, 'o-', color='green')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score by k')
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()

