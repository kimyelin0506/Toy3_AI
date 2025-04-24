import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import os
from sklearn.decomposition import PCA
"""
데이터 전처리:

데이터는 50% 샘플링되어 처리됩니다.

height(키) 컬럼은 피트와 인치 단위에서 센티미터로 변환됩니다.

weight(몸무게)는 파운드 단위에서 킬로그램으로 변환됩니다.

rating(평점)은 결측값을 중앙값으로 채우고, 5점 만점으로 반올림하여 rating_5 컬럼을 생성합니다.

body type(체형) 같은 범주형 데이터는 최빈값으로 결측값을 채웁니다.

원-핫 인코딩은 body type 컬럼에만 적용됩니다.

피처 스케일링:

연속형 데이터(몸무게, 키, 나이, 사이즈 등)는 MinMaxScaler를 사용하여 0과 1 사이로 정규화됩니다.

원-핫 인코딩된 body type 컬럼은 가중치를 적용하여 영향력을 조절합니다.

가중치 최적화:

PCA(주성분 분석)와 MI(상호 정보)를 기반으로 최적의 원-핫 인코딩 가중치를 계산합니다.

이를 통해 최적의 클러스터링을 위한 특성 선택과 가중치 설정을 자동으로 조정합니다.

KMeans 클러스터링:

최적의 K값(클러스터 수)을 찾기 위해 elbow method(엘보우 방법)와 silhouette score(실루엣 점수)를 사용합니다.

KMeans 클러스터링을 실행하고, 클러스터링의 성능을 실루엣 점수로 평가합니다.

결과 분석 및 시각화:

KMeans 클러스터링 결과를 PCA(2D)로 시각화하여 각 클러스터의 분포를 확인합니다.

각 클러스터의 특성 중심을 분석하여, 어떤 특성이 클러스터 분포에 영향을 미치는지 분석합니다.

클러스터링 성능 평가:

MI 기반 클러스터링과 PCA 기반 클러스터링의 실루엣 점수를 비교하여, 어떤 방법이 더 좋은 결과를 주는지 평가합니다.
"""
# 출력 디렉토리 생성
os.makedirs('plots', exist_ok=True)

# 데이터 로드
df = pd.read_csv('../renttherunway_data.csv')
print(f"전체 데이터 크기: {df.shape}")

# 👉 데이터 50% 샘플링
df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)

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
    data = data[data['fit'] == 'fit'].copy()

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
    # encoder_fit = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # fit_encoded = pd.DataFrame(encoder_fit.fit_transform(data[['fit']]),
    #                            columns=encoder_fit.get_feature_names_out(['fit']), index=data.index)

    encoder_body = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    body_encoded = pd.DataFrame(encoder_body.fit_transform(data[['body type']]),
                                columns=encoder_body.get_feature_names_out(['body type']), index=data.index)
    # data = pd.concat([data, fit_encoded, body_encoded], axis=1)
    data = pd.concat([data, body_encoded], axis=1)
    return data

# 스케일링 함수
def scaling_data(data, onehot_weight=0.2):
    scaler = MinMaxScaler()
    scaled_columns = ['weight_kg', 'height_cm', 'size', 'age']

    # 연속형 변수 스케일링
    for col in scaled_columns:
        data[f'{col}_scaled'] = scaler.fit_transform(data[[col]])

    # 원핫 인코딩 컬럼 찾기
    onehot_columns = [col for col in data.columns if col.startswith(('fit_', 'body type_'))]

    # 원핫 인코딩 컬럼에 가중치 적용
    for col in onehot_columns:
        data[col] = data[col] /7

    # 최종 feature 모음
    feature_columns = [f'{col}_scaled' for col in scaled_columns] + onehot_columns
    return data[feature_columns]

def find_auto_onehot_weight(train_df, optimal_k=20):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # 스케일링 + 연속형 변수 평균 계산
    scaled_data = train_df.copy()
    scaler = MinMaxScaler()
    scaled_columns = ['weight_kg', 'height_cm', 'size', 'age']

    for col in scaled_columns:
        scaled_data[f'{col}_scaled'] = scaler.fit_transform(scaled_data[[col]])

    # 연속형 스케일링된 컬럼들의 평균 계산
    feature_means = []
    for col in scaled_columns:
        mean_value = scaled_data[f'{col}_scaled'].mean()
        feature_means.append(mean_value)

    auto_onehot_weight = np.mean(feature_means)
    print(f"\n📈 자동 계산된 OneHot 가중치 (연속형 평균 기반): {auto_onehot_weight:.4f}")

    # 이제 이 가중치로 다시 스케일링
    onehot_columns = [col for col in scaled_data.columns if col.startswith(('fit_', 'body type_'))]
    for col in onehot_columns:
        scaled_data[col] = scaled_data[col] * auto_onehot_weight

    feature_columns = [f'{col}_scaled' for col in scaled_columns] + onehot_columns
    feature_df = scaled_data[feature_columns]

    # KMeans 학습
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++')
    labels = kmeans.fit_predict(feature_df)

    # 실루엣 점수 계산
    silhouette = silhouette_score(feature_df, labels)
    print(f"✅ 실루엣 점수 (auto weight): {silhouette:.4f}")

    return auto_onehot_weight, silhouette

# 데이터 분할 함수
def train_new_data_split(data, test_size, random_state):
    train_data, new_data = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"train 크기: {train_data.shape}, new 크기: {new_data.shape}")
    return train_data, new_data

# 최적의 k찾아서 적용
def find_optimal_k(train_features_df, k_min=2, k_max=30):
    wcss = []
    silhouette_scores = []

    for k in range(k_min, k_max + 1):
        """
        KMeans(n_clusters=optimal_k,
                random_state=42,
                n_init=10,
                init='k-means++',
                max_iter=500,
                algorithm='elkan',
                tol=1e-5)
        """
        # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
        kmeans = KMeans(n_clusters=k,
               random_state=42,
               n_init=10,
               init='k-means++',
               max_iter=500,
               algorithm='elkan',
               tol=1e-5)
        labels = kmeans.fit_predict(train_features_df)
        wcss.append(kmeans.inertia_)
        if k > 1:  # 실루엣 점수는 k=2 이상부터 의미 있음
            silhouette = silhouette_score(train_features_df, labels)
            silhouette_scores.append((k, silhouette))
            print(k,"실루엣 점수: ", silhouette)

    # 엘보우 플롯 저장
    plt.figure(figsize=(10, 6))
    plt.plot(range(k_min, k_max + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.savefig('plots/elbow_plot.png')
    plt.show()
    plt.close()

    # 실루엣 스코어가 가장 높은 k 선택
    best_k, best_silhouette = max(silhouette_scores, key=lambda x: x[1])
    print(f"\n✅ 자동으로 선택된 최적 K: {best_k} (Silhouette: {best_silhouette:.4f})")
    return best_k

# --- PCA 기반 가중치 계산 ---
def pca_based_onehot_weight(train_df, continuous_cols, onehot_cols):
    # 연속형, 범주형 구분
    cont_data = train_df[continuous_cols]
    onehot_data = train_df[onehot_cols]

    # 연속형 스케일링
    scaler = MinMaxScaler()
    cont_scaled = scaler.fit_transform(cont_data)

    # 원-핫은 그대로 사용
    data_combined = np.hstack([cont_scaled, onehot_data.values])

    # PCA 적용
    pca = PCA(n_components=min(data_combined.shape))
    pca.fit(data_combined)

    # 각 변수별 기여도
    loading_scores = np.abs(pca.components_).mean(axis=0)  # 각 feature별 평균 loading

    # 연속형, 범주형 분리
    cont_loading = loading_scores[:len(continuous_cols)].mean()
    onehot_loading = loading_scores[len(continuous_cols):].mean()

    # 비율 계산
    weight_ratio = cont_loading / onehot_loading
    print(f"연속형 평균 Loading: {cont_loading:.4f}, 범주형 평균 Loading: {onehot_loading:.4f}")
    print(f"추천하는 범주형 가중치 (PCA 기반): {weight_ratio:.4f}")

    return weight_ratio

# --- MI 기반 특성 선택 ---
def select_features_by_mi(X, y, top_n=10):
    """
    Mutual Information을 사용해 가장 중요한 feature n개 선택
    """
    # Mutual Information 계산
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values(by='mi_score', ascending=False)

    print("\n📊 Mutual Information 상위 Feature:")
    print(mi_df.head(top_n))

    selected_features = mi_df['feature'].iloc[:top_n].tolist()
    return selected_features

# --- KMeans 클러스터링 함수 ---
def apply_kmeans_with_silhouette(train_features_df, optimal_k, features_selected=None):
    # 최적 클러스터링 수행
    if features_selected:
        train_features_df = train_features_df[features_selected]

    kmeans = KMeans(n_clusters=optimal_k,
                    random_state=42,
                    n_init=10,
                    init='k-means++',
                    max_iter=500,
                    algorithm='elkan',
                    tol=1e-5)
    train_df['cluster'] = kmeans.fit_predict(train_features_df)

    # 실루엣 점수 확인
    silhouette = silhouette_score(train_features_df, train_df['cluster'])
    print(f"\n선택된 Feature 기반 학습 데이터 실루엣 점수: {silhouette:.3f}")

    return kmeans, silhouette

# --- PCA 기반과 MI 기반을 모두 적용하는 함수 ---
def apply_pca_and_mi_based_clustering(train_df, continuous_cols, onehot_cols):
    # 1. PCA 기반 OneHot 가중치 계산
    pca_weight = pca_based_onehot_weight(train_df, continuous_cols, onehot_cols)

    # 2. MI 기반 특성 선택
    selected_features_mi = select_features_by_mi(train_df[continuous_cols + onehot_cols], train_df['cluster_initial'], top_n=10)

    # 3. PCA 가중치 적용
    train_features_df_pca = scaling_data(train_df, onehot_weight=pca_weight)

    # 4. MI 특성만 적용하여 KMeans 수행
    optimal_k_mi = find_optimal_k(train_features_df_pca, k_min=8, k_max=21)
    kmeans_mi, silhouette_mi = apply_kmeans_with_silhouette(train_features_df_pca, optimal_k_mi, selected_features_mi)

    # 5. PCA 기반 KMeans 수행
    optimal_k_pca = find_optimal_k(train_features_df_pca, k_min=6, k_max=21)
    kmeans_pca, silhouette_pca = apply_kmeans_with_silhouette(train_features_df_pca, optimal_k_pca)

    return kmeans_mi, silhouette_mi, kmeans_pca, silhouette_pca

# --- 리팩토링 실행 흐름 시작 ---
# 1. 데이터 분할 (원본에서 먼저!)
train_df_raw, new_df_raw = train_new_data_split(df, test_size=0.01, random_state=42)

# 2. 각각 전처리
train_df = preprocessing_data(train_df_raw)
new_df = preprocessing_data(new_df_raw)

# 이 auto_weight를 scaling_data()에 넘겨서 최종 데이터 준비
train_features_df = scaling_data(train_df, onehot_weight=0.7989)
new_features_df = scaling_data(new_df, onehot_weight=0.7989)

# best_weight를 scaling_data()에 넘겨서 최종 데이터 준비하기
# train_features_df = scaling_data(train_df, onehot_weight=best_weight)
# new_features_df = scaling_data(new_df, onehot_weight=best_weight)

# 4. 최적 클러스터 찾기 (자동)
optimal_k = find_optimal_k(train_features_df, k_min=6, k_max=21)
# optimal_k = 21

# 5. K-Means 클러스터링
# optimal_k = 20
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, init='k-means++')
# train_df['cluster'] = kmeans.fit_predict(train_features_df)

# --- 최종 실행 ---
# continuous_cols와 onehot_cols 정의
continuous_cols = ['height_cm', 'weight_kg', 'rating_5', 'age', 'size']
onehot_cols = [col for col in train_df.columns if col.startswith(('fit_', 'body type_'))]

# PCA 및 MI 기반 클러스터링 적용
kmeans_mi, silhouette_mi, kmeans_pca, silhouette_pca = apply_pca_and_mi_based_clustering(train_df, continuous_cols, onehot_cols)

# 5. K-Means 클러스터링
def apply_kmeans_clustering(train_df, features_df, optimal_k, kmeans_init='k-means++', n_init=10):
    """
    MI 및 PCA 기반으로 클러스터링을 적용한 후, 클러스터링 결과를 반환
    :param train_df: 학습 데이터프레임
    :param features_df: 특성 데이터프레임 (PCA와 MI 기반)
    :param optimal_k: 최적 클러스터 개수
    :param kmeans_init: KMeans 초기화 방식
    :param n_init: KMeans n_init 값
    :return: 학습된 KMeans 모델, 클러스터 레이블
    """
    kmeans = KMeans(n_clusters=optimal_k,
                    random_state=42,
                    n_init=n_init,
                    init=kmeans_init,
                    max_iter=500,
                    algorithm='elkan',
                    tol=1e-5)

    # KMeans 클러스터링
    train_df['cluster'] = kmeans.fit_predict(features_df)

    # 실루엣 점수 확인
    silhouette = silhouette_score(features_df, train_df['cluster'])
    print(f"학습 데이터 실루엣 점수: {silhouette:.3f}")

    return kmeans, silhouette


# 6. 학습 데이터에 대한 KMeans 모델 학습
kmeans_mi_final, silhouette_mi_final = apply_kmeans_clustering(train_df, new_features_df, optimal_k)
kmeans_pca_final, silhouette_pca_final = apply_kmeans_clustering(train_df, train_features_df, optimal_k)

# 7. 클러스터링 결과 분석 (MI, PCA 기반 결과 비교)
print(f"MI 기반 클러스터링 실루엣 점수: {silhouette_mi_final:.3f}")
print(f"PCA 기반 클러스터링 실루엣 점수: {silhouette_pca_final:.3f}")


# 8. 클러스터링 결과 시각화 (예: PCA 기반 클러스터링 결과 2D 시각화)
def plot_cluster_results(features_df, cluster_labels, title="Cluster Plot"):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_df)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=cluster_labels, palette='viridis', s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# PCA 기반 클러스터링 결과 시각화
plot_cluster_results(train_features_df, kmeans_pca_final.labels_, title="PCA 기반 클러스터링 결과")

# MI 기반 클러스터링 결과 시각화
plot_cluster_results(new_features_df, kmeans_mi_final.labels_, title="MI 기반 클러스터링 결과")


# 9. 클러스터별 특성 분석
def analyze_cluster_centroids(kmeans_model, feature_names):
    print("\n클러스터 중심 분석:")
    centroids = kmeans_model.cluster_centers_
    cluster_centroids_df = pd.DataFrame(centroids, columns=feature_names)
    print(cluster_centroids_df)


# MI 기반 클러스터링 centroids 분석
analyze_cluster_centroids(kmeans_mi_final, continuous_cols + onehot_cols)

# PCA 기반 클러스터링 centroids 분석
analyze_cluster_centroids(kmeans_pca_final, continuous_cols + onehot_cols)