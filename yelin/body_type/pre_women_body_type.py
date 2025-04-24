# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# 2. 데이터 불러오기
# data = pd.read_csv('women body syn.csv')
data = pd.read_csv('women_body_with_bmi.csv')
# 3. 전처리

## 3-1. 컬럼명 공백 제거
data.columns = [col.strip() for col in data.columns]
print('Body Shape index(unique)', data['Body Shape Index'].unique())
print('Body Shape Index (count per unique value, sorted)')
print(data['Body Shape Index'].value_counts().sort_index())
print()

print('샘플 개수: ', len(data))
print('특성 정보(컬럼명/타입)')
print(data.dtypes)
## 3-2. 결측치 제거
data = data.dropna()

## 3-3. Gender, Cup Size 인코딩
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])  # Male=1, Female=0

le_cup = LabelEncoder()
data['Cup Size'] = le_cup.fit_transform(data['Cup Size'])  # 컵 사이즈 문자형 인코딩

# (2) 컬럼명 정리
data.columns = [col.strip() for col in data.columns]

# (3) 'body_type' 라벨 새로 만들기
# 간단한 예시 기준 (나중에 너가 바꿔도 됨)
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


data['body_type'] = data.apply(classify_body_type, axis=1)
data = data.drop('Body Shape Index', axis=1)

print('샘플 개수: ', len(data))
print('특성 정보(컬럼명/타입)')
print(data.dtypes)
# (4) 이후에야
X = data.drop('body_type', axis=1)
y = data['body_type']

## 3-5. y 라벨 인코딩
le_body = LabelEncoder()
y_encoded = le_body.fit_transform(y)

## 3-6. 데이터 스케일링 (표준화)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. 학습용 데이터, 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(X.dtypes)

# 5. 전처리된 데이터 시각화
"""
(1) 키/몸무게/허리/힙/가슴 치수 분포 (전체적인 생김새 파악)
(2) Body Type별 키/몸무게 차이 (Body Type 간 차이 시각화)
(3) 특성 간 상관성 (어떤 특성끼리 연관성이 있는지)
(4) PCA 축소 (고차원 공간을 2D로 압축해서 데이터 분포를 한눈에 보기)
"""
# print(data.columns)
# # (1) 수치형 특성 분포 보기
# # Body Type 인덱스 매핑
# body_type_mapping = {i: name for i, name in enumerate(sorted(data['body_type'].unique()))}
# print("Body Type Index Mapping:")
# for idx, name in body_type_mapping.items():
#     print(f"{idx}: {name}")
#
# # 데이터에 인덱스 컬럼 추가
# data['body_type_idx'] = data['body_type'].map({v: k for k, v in body_type_mapping.items()})
# # numeric_cols = ['Height', 'Weight', 'Bust/Chest', 'Waist', 'Hips', 'BMI', 'Cup Size']
# numeric_cols = ['Gender', 'Weight',  'Waist', 'Hips', 'Bust/Chest', 'Height', 'Cup Size', 'BMI', 'body_type_idx']
#
#
# plt.figure(figsize=(15, 8))
# for i, col in enumerate(numeric_cols, 1):
#     plt.subplot(2, 5, i)
#     sns.histplot(data[col], kde=True, color='skyblue')
#     plt.title(f'Distribution of {col}')
# plt.tight_layout()
# plt.show()
# import numpy as np
#
# # (5) 모든 수치형 컬럼별 Body Type 박스플롯
# plt.figure(figsize=(20, 25))  # 전체 figure 크기 크게 잡기
#
# for i, col in enumerate(numeric_cols, 1):
#     plt.subplot(4, 3, i)  # 4행 3열 그리드 (총 12개 plot)
#     sns.boxplot(x='body_type', y=col, data=data)
#     plt.title(f'{col} Distribution by Body Type')
#     plt.xticks(rotation=45)
#
#     # === 이상치 개수 계산 ===
#     for body_type_val in data['body_type'].unique():
#         subset = data[data['body_type'] == body_type_val][col]
#         Q1 = subset.quantile(0.25)
#         Q3 = subset.quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
#         print(f"[{col}] Body Type '{body_type_val}': 이상치 개수 = {len(outliers)}")
#
# plt.tight_layout()
# plt.show()
#
# # (0) Body Type 인덱스 매핑
#
# # 박스플롯 그리기
# # 1. Body Shape Index 제외
# numeric_cols_v2 = [col for col in numeric_cols if col != 'Body Shape Index']
#
# # 2개 그룹으로 나누기
# half = len(numeric_cols_v2) // 2
# split_numeric_cols = [numeric_cols_v2[:half], numeric_cols_v2[half:]]
#
# for group_num, cols in enumerate(split_numeric_cols, 1):
#     fig, axes = plt.subplots(2, 2, figsize=(15, 7))  # (16, 12)로 좀 더 현실적인 크기
#     plt.subplots_adjust(hspace=5, wspace=0.4)
#
#     axes = axes.flatten()
#
#     for i, col in enumerate(cols):
#         sns.boxplot(x='body_type_idx', y=col, data=data, ax=axes[i])
#         axes[i].set_title(f'{col} Distribution by Body Type Index', fontsize=15, pad=15)
#         axes[i].set_xlabel('Body Type Index', fontsize=12)
#         axes[i].set_ylabel(col, fontsize=12)
#         axes[i].set_xticks(list(body_type_mapping.keys()))
#         axes[i].set_xticklabels(list(body_type_mapping.keys()), fontsize=10)
#         axes[i].tick_params(axis='y', labelsize=10)
#
#         # === 이상치 개수 계산 ===
#         for idx in data['body_type_idx'].unique():
#             subset = data[data['body_type_idx'] == idx][col]
#             Q1 = subset.quantile(0.25)
#             Q3 = subset.quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             outliers = subset[(subset < lower_bound) | (subset > upper_bound)]
#             print(f"[{col}] Body Type Index {idx}: 이상치 개수 = {len(outliers)}")
#
#     # 남는 subplot 삭제
#     for j in range(len(cols), len(axes)):
#         fig.delaxes(axes[j])
#
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.show()
#
#
#
# # (2) Body Type별 Height, Weight 분포
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='body_type', y='Height', data=data)
# plt.title('Height Distribution by Body Type')
# plt.xticks(rotation=45)
# plt.show()
#
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='body_type', y='Weight', data=data)
# plt.title('Weight Distribution by Body Type')
# plt.xticks(rotation=45)
# plt.show()
#
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='body_type', y='BMI', data=data)
# plt.title('BMI Distribution by Body Type')
# plt.xticks(rotation=45)
# plt.show()
#
# # (3) 상관계수 히트맵
# corr = data[numeric_cols].corr()
#
# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Feature Correlation Heatmap')
# plt.show()
#
# # (4) PCA 2D 시각화 (Optional)
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# plt.figure(figsize=(10,8))
# scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_encoded, cmap='tab10', alpha=0.7)
# plt.legend(*scatter.legend_elements(), title="Body Type", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.title('PCA Projection (2D)')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.grid(True)
# plt.show()
#
