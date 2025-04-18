import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import os

# 출력 디렉토리 생성
os.makedirs('plots', exist_ok=True)

# 데이터 로드
df = pd.read_csv('../renttherunway_data.csv')
print(f"전체 데이터 크기: {df.shape}")


# 키를 cm로 변환하는 함수
def height_to_cm(height):
    if pd.isna(height):
        return np.nan
    try:
        feet, inches = height.split("'")
        inches = inches.replace('"', '').strip()
        return int(feet) * 30.48 + int(inches) * 2.54
    except:
        return np.nan

# 데이터 전처리 함수
def preprocessing_data(data):
    selected_columns = ['user_id', 'item_id', 'fit', 'weight', 'rating', 'body type', 'height', 'size', 'age']
    data = data[selected_columns].copy()

    data['height_cm'] = data['height'].apply(height_to_cm)
    data['weight'] = data['weight'].astype(str).str.replace('lbs', '').str.strip()
    data['weight'] = pd.to_numeric(data['weight'], errors='coerce')
    data['weight_kg'] = data['weight'] * 0.453592

    data['rating'] = pd.to_numeric(data['rating'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    rating_median = data['rating'].median()
    data['rating'] = data['rating'].fillna(rating_median)
    data['rating_5'] = (data['rating'] / 2).round().astype(int).clip(1, 5)

    numeric_columns = ['height_cm', 'weight_kg', 'rating_5', 'age', 'size']
    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].mean())

    categorical_columns = ['fit', 'body type']
    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])

    # 원-핫 인코딩
    encoder_fit = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    fit_encoded = pd.DataFrame(encoder_fit.fit_transform(data[['fit']]),
                               columns=encoder_fit.get_feature_names_out(['fit']), index=data.index)

    encoder_body = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    body_encoded = pd.DataFrame(encoder_body.fit_transform(data[['body type']]),
                                columns=encoder_body.get_feature_names_out(['body type']), index=data.index)

    data = pd.concat([data, fit_encoded, body_encoded], axis=1)
    return data

# 스케일링 함수
def scaling_data(data):
    scaler = MinMaxScaler()
    scaled_columns = ['weight_kg', 'height_cm', 'size', 'age']
    for col in scaled_columns:
        data[f'{col}_scaled'] = scaler.fit_transform(data[[col]])

    feature_columns = [f'{col}_scaled' for col in scaled_columns] + \
                      [col for col in data.columns if col.startswith(('fit_', 'body type_'))]
    return data[feature_columns]

# 데이터 분할 함수
def train_new_data_split(data, test_size, random_state):
    train_data, new_data = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"train 크기: {train_data.shape}, new 크기: {new_data.shape}")
    return train_data, new_data

# --- 리팩토링 실행 흐름 시작 ---
"""
1. 미리 전처리하고 나누는 경우

➡️ 전처리(결측치 채우기, 스케일링 등)를 전체 데이터에 적용하고 나눔.

➡️ 새 데이터가 훈련 데이터 정보를 일부 '엿본' 상태가 될 수 있음.

➡️ 특히 평균, 중앙값, 최빈값을 채울 때 전체 평균을 사용하면,

실제 새로운 데이터에서는 사용할 수 없는 정보를 이미 써버린 게 돼.

(→ 현실에서는 실제로 "새 데이터"는 우리가 아직 모르는 상태로 들어오잖아?
그런 상황을 연습하려고 이렇게 엄격하게 관리하는 거야.)

2. 각각 전처리하는 경우 (내가 한 방법)

➡️ train/test 나눈 다음, train에서만 평균/중앙값/최빈값 계산.

➡️ test(new) 데이터는 훈련 데이터로 만들어진 기준에 맞춰 적용.

➡️ 진짜로 "모르는 데이터"를 다루는 것처럼 훈련하고 평가할 수 있어.
"""
# 1. 데이터 분할 (원본에서 먼저!)
train_df_raw, new_df_raw = train_new_data_split(df, test_size=0.1, random_state=42)

# 2. 각각 전처리
train_df = preprocessing_data(train_df_raw)
new_df = preprocessing_data(new_df_raw)

# 3. 각각 스케일링
train_features_df = scaling_data(train_df)
new_features_df = scaling_data(new_df)

# 4. 엘보우 기법으로 최적 클러스터 찾기
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(train_features_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.savefig('plots/elbow_plot.png')
plt.show()
plt.close()

# 5. K-Means 클러스터링
optimal_k = 20
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
train_df['cluster'] = kmeans.fit_predict(train_features_df)

# 6. 학습 데이터 실루엣 점수
train_silhouette = silhouette_score(train_features_df, train_df['cluster'])
print(f"\n학습 데이터 실루엣 점수: {train_silhouette:.3f}")

# 7. 군집별 특성 분석
original_columns = ['weight_kg', 'height_cm', 'size', 'age', 'rating_5']
print("\n학습 데이터 군집별 평균:")
print(train_df.groupby('cluster')[original_columns].mean())

# 8. 추천 상품 선정
cluster_items = train_df.groupby('cluster').apply(
    lambda x: x.nlargest(5, 'rating_5')[['item_id', 'rating_5']].to_dict('records')
)

print("\n군집별 상위 5개 추천 상품:")
for cluster, items in cluster_items.items():
    print(f"Cluster {cluster}:")
    for item in items:
        print(f"  Item ID: {item['item_id']}, Rating: {item['rating_5']}")

# 9. 새로운 데이터 군집 예측
new_df['cluster'] = kmeans.predict(new_features_df)

print("\n새로운 데이터 군집 분포:")
print(new_df['cluster'].value_counts())

# 10. 새로운 데이터 실루엣 점수
new_silhouette = silhouette_score(new_features_df, new_df['cluster'])
print(f"\n새로운 데이터 실루엣 점수: {new_silhouette:.3f}")


# 11. Precision, Recall, F1 평가
def evaluate_recommendations(df, cluster_items, K=5):
    precision_scores = []
    recall_scores = []

    for cluster in df['cluster'].unique():
        real_items = set(df[(df['cluster'] == cluster) & (df['rating_5'] >= 4)]['item_id'])
        recommended_items = set(item['item_id'] for item in cluster_items.get(cluster, []))
        hits = len(real_items & recommended_items)
        precision = hits / K if K > 0 else 0
        recall = hits / len(real_items) if real_items else 0
        precision_scores.append(precision)
        recall_scores.append(recall)

    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_scores, recall_scores)]
    return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)

train_precision, train_recall, train_f1 = evaluate_recommendations(train_df, cluster_items)
new_precision, new_recall, new_f1 = evaluate_recommendations(new_df, cluster_items)

print(f"\n학습 데이터 Precision@5: {train_precision:.3f}, Recall@5: {train_recall:.3f}, F1 Score: {train_f1:.3f}")
print(f"새로운 데이터 Precision@5: {new_precision:.3f}, Recall@5: {new_recall:.3f}, F1 Score: {new_f1:.3f}")

# 12. 새로운 데이터 추천 품질 (Feedback Score)
feedback_scores = []
for cluster in new_df['cluster'].unique():
    recommended_items = set(item['item_id'] for item in cluster_items.get(cluster, []))
    cluster_data = new_df[(new_df['cluster'] == cluster) & (new_df['item_id'].isin(recommended_items))]
    if not cluster_data.empty:
        feedback_score = (cluster_data['rating_5'] >= 4).mean()
        feedback_scores.append(feedback_score)
    else:
        feedback_scores.append(0)

print(f"\n새로운 데이터 추천 상품의 높은 평점 비율 (평균): {np.mean(feedback_scores):.3f}")

# 13. 새로운 데이터 추천 상품 출력
print("\n새로운 데이터에 대한 군집별 추천 상품:")
for cluster in new_df['cluster'].unique():
    print(f"Cluster {cluster} 추천 상품:")
    for item in cluster_items.get(cluster, []):
        print(f"  Item ID: {item['item_id']}, Rating: {item['rating_5']}")
