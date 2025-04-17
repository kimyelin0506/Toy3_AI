import pandas as pd

# 데이터 불러오기
data = pd.read_csv('../Body Measurements.csv')  # 파일명 맞게 수정해줘

# 컬럼명 바꾸기(빈공백 제거)
data.columns = data.columns.str.replace(' ', '')

"""1. body_type 레이블 생성"""
# 레이블 생성 함수
def classify_body_type(row):
    chest = row['ChestWidth']
    waist = row['Waist']
    hips = row['Hips']
    shoulder = row['ShoulderWidth']
    height = row['TotalHeight']

    chest_waist_ratio = chest / waist
    waist_hip_ratio = waist / hips
    shoulder_waist_ratio = shoulder / waist
    waist_height_ratio = waist / height

    if height < 63:
        return 'petite'
    if abs(chest - hips) <= 2 and waist < 0.75 * chest:
        return 'hourglass'
    if chest > hips and waist > 0.8 * chest:
        return 'apple'
    if hips > chest and waist < 0.8 * hips:
        return 'pear'
    if shoulder > chest and abs(chest - hips) <= 3 and waist > 0.75 * chest:
        return 'athletic'
    if chest > hips + 3:
        return 'full bust'
    return 'straight & narrow'


# 적용
data['body_type'] = data.apply(classify_body_type, axis=1)

"""2. 전처리 (X, y 나누기 + 스케일링)"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 특성과 타겟 분리
X = data.drop(['body_type'], axis=1)
y = data['body_type']

# 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.2, random_state=42, stratify=None)

"""3. 모델 학습 (랜덤포레스트)"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 모델 선언
model = RandomForestClassifier(random_state=42)

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(classification_report(y_test, y_pred))

"""confusion matrix 시각화"""
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
