# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
"""
1	기본 전처리하고, Cup Size, Gender 인코딩
2	classify_body_type()로 개별 사람 체형 구하기
3	K-means로 군집화
4	군집마다 가장 흔한 body type 찾아서 대표로 지정
5	각 사람에게 '나의 체형'과 '군집 대표 체형' 둘 다 보여줌
"""
# 2. 데이터 불러오기 및 기본 전처리
data = pd.read_csv('women body syn.csv')
data.columns = [col.strip() for col in data.columns]
data = data.dropna()

# Label Encoding
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

le_cup = LabelEncoder()
data['Cup Size'] = le_cup.fit_transform(data['Cup Size'])

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
    elif height < 62:  # 157cm 이하
        return 'body_type_petite'
    elif abs(bust - hips) < 2 and waist / hips >= 0.8:
        return 'body_type_straight & narrow'
    else:
        return 'body_type_athletic'

# 4. 스케일링
X = data.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 군집화 (k-means 사용)
n_clusters = 5  # 너가 원하는 군집 개수로 수정 가능
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

data['cluster'] = clusters  # 군집 결과 붙이기

# 6. 각 사람별로 body_type 계산
data['individual_body_type'] = data.apply(classify_body_type, axis=1)

# 7. 군집별 대표 body_type 결정
cluster_to_bodytype = {}

for cluster_id in sorted(data['cluster'].unique()):
    subset = data[data['cluster'] == cluster_id]
    most_common_type = subset['individual_body_type'].mode()[0]  # 가장 많은 타입
    cluster_to_bodytype[cluster_id] = most_common_type

# 8. 최종 결과 보기
data['representative_body_type'] = data['cluster'].map(cluster_to_bodytype)

print(data[['Bust/Chest', 'Waist', 'Hips', 'Height', 'cluster', 'individual_body_type', 'representative_body_type']].head())
