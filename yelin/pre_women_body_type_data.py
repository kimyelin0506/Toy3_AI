# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 2. 데이터 불러오기
data = pd.read_csv('../women body syn.csv')

# 3. 전처리

## 3-1. 컬럼명 공백 제거
data.columns = [col.strip() for col in data.columns]

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
    elif height < 62:  # 157cm 이하
        return 'body_type_petite'
    elif abs(bust - hips) < 2 and waist / hips >= 0.8:
        return 'body_type_straight & narrow'
    else:
        return 'body_type_athletic'


data['body_type'] = data.apply(classify_body_type, axis=1)

# (4) 이후에야
X = data.drop('body_type', axis=1)
y = data['body_type']

## 3-5. y 라벨 인코딩
le_body = LabelEncoder()
y_encoded = le_body.fit_transform(y)

## 3-6. 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 학습용 데이터, 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(X.dtypes)
