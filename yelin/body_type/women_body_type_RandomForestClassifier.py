# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pre_women_body_type_data import X_train, X_test, y_train, y_test, le_body



# 5-1. 머신러닝 모델 (랜덤포레스트) 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 5-2. 랜덤포레스트 평가
rf_preds = rf_model.predict(X_test)
print("Random Forest Classification Report")
print(classification_report(y_test, rf_preds, target_names=le_body.classes_))

# 5-3. 랜덤포레스트 혼동행렬 시각화
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=le_body.classes_, yticklabels=le_body.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


