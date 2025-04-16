import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 데이터 시각화를 위한 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('../renttherunway_data.csv')
    
# 선택한 컬럼만 사용
selected_columns = ['user_id', 'item_id', 'fit', 'weight', 'rating', 'body type', 'height', 'size', 'age', 'bust size']
df = df[selected_columns]


# 결측치 비율 확인
print("결측치 비율 (%):")
print(df.isna().mean() * 100)

# 1. 단위 변환
# height: feet/inch → cm 변환
def height_to_cm(height):
    if pd.isna(height):
        return np.nan
    try:
        feet, inches = height.split("'")
        inches = inches.replace('"', '').strip()
        return int(feet) * 30.48 + int(inches) * 2.54
    except:
        return np.nan

df['height_cm'] = df['height'].apply(height_to_cm)

# weight: lbs → kg 변환 (먼저 숫자만 추출)
df['weight'] = df['weight'].astype(str).str.replace('lbs', '').str.strip()
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['weight_kg'] = df['weight'] * 0.453592

# rating: 1~10 → 1~5 변환
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_5'] = ((df['rating'] - 1) / 2 + 1).round().astype(int)

# 2. 결측치 처리
# 먼저 모든 숫자형 컬럼의 결측치를 평균값으로 대체
numeric_columns = ['height_cm', 'weight_kg', 'rating_5', 'age', 'size']
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# 범주형 컬럼의 결측치를 최빈값으로 대체
categorical_columns = ['bust size', 'body type']
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 결측치 처리 후 확인
print("\n결측치 처리 후:")
print(df.isna().sum())

# 3. 원-핫 인코딩
# fit: 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
fit_encoded = encoder.fit_transform(df[['fit']])
fit_encoded_df = pd.DataFrame(fit_encoded, columns=encoder.get_feature_names_out(['fit']))

# bust size: 상위 15개 + 'other'
top_15_bust = df['bust size'].value_counts().head(15).index
df['bust size_grouped'] = df['bust size'].apply(lambda x: x if x in top_15_bust else 'other')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
bust_encoded = encoder.fit_transform(df[['bust size_grouped']])
bust_encoded_df = pd.DataFrame(bust_encoded, columns=encoder.get_feature_names_out(['bust size_grouped']))

# body type: 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
body_encoded = encoder.fit_transform(df[['body type']])
body_encoded_df = pd.DataFrame(body_encoded, columns=encoder.get_feature_names_out(['body type']))

# category: 상위 10개 + 'other'
top_10_categories = df['category'].value_counts().head(10).index
df['category_grouped'] = df['category'].apply(lambda x: x if x in top_10_categories else 'other')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
category_encoded = encoder.fit_transform(df[['category_grouped']])
category_encoded_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['category_grouped']))

# 4. 스케일링
scaler = MinMaxScaler()
df['weight_kg_scaled'] = scaler.fit_transform(df[['weight_kg']])
df['rating_scaled'] = scaler.fit_transform(df[['rating_5']])
df['height_cm_scaled'] = scaler.fit_transform(df[['height_cm']])
df['size_scaled'] = scaler.fit_transform(df[['size']])
df['age_scaled'] = scaler.fit_transform(df[['age']])

# 모든 처리된 특성 결합
df = pd.concat([df, fit_encoded_df, bust_encoded_df, body_encoded_df, category_encoded_df], axis=1)

# 학습용 피처 선택 (user_id, item_id 제외)
feature_columns = [col for col in df.columns if col not in ['user_id', 'item_id', 'fit', 'weight', 'rating', 'body type', 'category', 'height', 'size', 'age', 'bust size', 'bust size_grouped', 'category_grouped', 'rating_5', 'weight_kg']]
features_df = df[feature_columns]

# csv 저장
df.to_csv('preprocessed_data.csv', index=False)

# 피처 컬럼 확인
print("\n학습용 피처 컬럼:", feature_columns)

# 범주형 분포
for col in ['bust size_grouped', 'body type', 'category_grouped', 'fit']:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=45)
    plt.show()

# 수치형 분포
for col in ['weight_kg_scaled', 'height_cm_scaled', 'age_scaled', 'rating_scaled', 'size_scaled']:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], bins=30)
    plt.title(f'{col} Distribution')
    plt.show()