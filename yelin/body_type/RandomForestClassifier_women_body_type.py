# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pre_women_body_type import X_train, X_test, y_train, y_test, le_body
"""
"Random Forest" 모델
각각 다른 Decision Tree 여러 개를 학습
결과를 다수결 투표로 결정
과적합 방지에 강하고, 분류 성능도 꽤 좋음
(단점: 트리 개수가 많으면 무겁고 해석이 어려워질 수 있다는 것.)
"""

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


