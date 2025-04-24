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
"""
주요 지표 의미:
Precision (정밀도):
예측한 것 중에 정답인 비율
(ex. "hourglass"라고 예측한 것들 중 진짜 hourglass 비율)

Recall (재현율):
실제 정답 중에 예측에 성공한 비율
(ex. 실제 hourglass인 것들 중 얼마나 잘 찾아냈는지)

F1-score:
Precision과 Recall의 조화 평균 (균형 지표)
(높을수록 좋은 성능)

Support:
각 클래스별 테스트 데이터 개수
(ex. full bust는 2186개, hourglass는 19개)
"""
"""
Random Forest Classification Report
                             precision    recall  f1-score   support

            body_type_apple       0.93      0.97      0.95       663
         body_type_athletic       0.93      0.96      0.95       304
        body_type_full bust       0.98      0.99      0.98      2186
        body_type_hourglass       0.00      0.00      0.00        19
             body_type_pear       0.97      0.97      0.97       668
           body_type_petite       0.95      0.92      0.94       128
body_type_straight & narrow       0.00      0.00      0.00        32

                   accuracy                           0.97      4000
                  macro avg       0.68      0.69      0.68      4000
               weighted avg       0.95      0.97      0.96      4000
               
(2) 결과 해석:

body_type_full bust, body_type_apple, body_type_pear, body_type_athletic, body_type_petite
→ precision, recall, f1-score 다 높다. (0.94~0.98)
→ 이 클래스들은 모델이 잘 맞췄다는 뜻!

body_type_hourglass, body_type_straight & narrow
→ precision, recall, f1-score가 0.00...
→ 이 클래스들은 거의 못 맞췄다는 의미야. (샘플 수가 매우 적거나 구분이 어려운 케이스)

Accuracy (전체 정확도):
97%로 꽤 높음. (하지만 일부 소수 클래스는 무시된 효과일 수 있음)

macro avg:
모든 클래스를 동일 가중치로 평균낸 것. (클래스 imbalance 고려)

macro f1-score가 0.68 → 일부 클래스를 잘 못 맞춘 영향.

weighted avg:
클래스별 sample 수를 고려해서 평균낸 것. (전체 평가 느낌)

weighted f1-score가 0.96 → 데이터셋 전체적으로 보면 매우 잘 맞춘 것.
"""
"""
2. Confusion Matrix 시각화 해석
Confusion Matrix는:

x축: 예측된 클래스

y축: 실제 정답 클래스

(y,x) 위치의 값: 실제 y 클래스를 x 클래스로 예측한 개수

(1) diagonal (대각선):

대각선 숫자가 크면 클수록 좋은 모델!

대각선에 많은 숫자가 몰려 있으면 예측을 잘 했다는 의미야.

(2) off-diagonal (대각선 외):

대각선 바깥쪽에 숫자가 크면 → 모델이 헷갈려서 틀렸다는 뜻.

(3) 네 시각화에서 볼 수 있는 점:

full bust, apple, pear 같은 클래스는 대각선 숫자가 큼 → 잘 맞춘 거야.

hourglass, straight & narrow는 대각선에 거의 숫자가 없음 → 거의 예측 실패.

특히 hourglass는 샘플이 19개밖에 없었는데 아예 못 맞췄어.
"""
"""
3. 전체 요약
데이터 imbalance(특정 클래스 개수 차이)가 심함.
→ 그래서 일부 소수 클래스들은 예측이 거의 안 됨.

다수 클래스를 중심으로는 예측 성능이 매우 좋음.

confusion matrix 시각화를 통해 어떤 클래스에서 성능이 좋은지/나쁜지 직관적으로 볼 수 있음.

따라서: 다수 클래스에는 강하고, 소수 클래스는 추가 보완 필요!
"""
# 5-1. 머신러닝 모델 (랜덤포레스트) 학습
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 5-2. 랜덤포레스트 평가
rf_preds = rf_model.predict(X_test)
print("Random Forest Classification Report")
print(classification_report(y_test, rf_preds, target_names=le_body.classes_))

# 5-3. 랜덤포레스트 혼동행렬 시각화
plt.figure(figsize=(20,10))

# 축 레이블 깔끔하게 정리
xticklabels = [label.replace('body_type_', '') for label in le_body.classes_]
yticklabels = [label.replace('body_type_', '') for label in le_body.classes_]

sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues',
            xticklabels=xticklabels, yticklabels=yticklabels)

plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



