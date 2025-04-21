# 1. 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from pre_women_body_type import X_train, X_test, y_train, y_test, y_encoded, le_body
"""
[딥러닝 신경망 모델(MLP)을 만들어서, 사람 체형(body type)을 분류하는 것]
<전체 흐름 요약>
(1) 필요한 라이브러리 불러오기
- numpy: 수치 계산용
- matplotlib/seaborn: 그래프 그리기
- sklearn: 평가 지표 (confusion matrix, classification report)
- keras: 딥러닝 모델 구축
- 미리 전처리해놓은 X_train, X_test 등 데이터셋 가져옴.

(2) 신경망 모델(MLP) 만들기
- Sequential 모델: 한 층씩 순서대로 쌓는 방식
    - 첫 번째 Dense 층: 128개의 뉴런, ReLU 활성화
    - Dropout(0.3): 30% 랜덤으로 노드 끄기 (과적합 방지)
    - 두 번째 Dense 층: 64개 뉴런, ReLU
    - 또 Dropout
    - 마지막 Dense 층: 클래스 수만큼 출력, Softmax (다중 클래스 분류)
    - 요약: 심플한 MLP 신경망
- 모델 컴파일
    - adam: 최적화 알고리즘
    - sparse_categorical_crossentropy: 다중 클래스 분류용 loss 함수 (레이블이 one-hot encoding이 아니라 정수형일 때 사용)
    - metrics=['accuracy']: 정확도 확인

(3) 모델 학습 (fit)
- early_stopping: validation loss가 5번 연속 좋아지지 않으면 학습 중단 (과적합 방지)
- history: 학습 과정 저장 (loss/accuracy 기록)

(4) 모델 평가 (confusion matrix, classification report, F1 score)
- 예측 결과 확률 → 가장 높은 확률 클래스를 선택.
- precision / recall / f1-score 출력해서 성능 평가.
- confusion matrix 시각화: 어떤 클래스를 잘 맞췄고, 헷갈렸는지 시각적으로 확인.
- weighted F1 score 계산해서 전체적인 모델 성능 확인.

(5) 결과 시각화 (loss/accuracy 그래프)
- Overfitting이 있는지/성공적으로 학습했는지 시각적으로 확인.
"""
"""
1. 왜 Dropout을 2번이나 넣었을까?
👉 과적합(Overfitting)을 방지하려고.
Dropout은 학습할 때 랜덤하게 일부 뉴런을 꺼버려서 모델이 특정 뉴런에 과하게 의존하지 못하게 막줌
첫 번째 Dense(128 뉴런) 뒤에서 한 번
두 번째 Dense(64 뉴런) 뒤에서 한 번
2번 넣은 이유는: "두 개의 큰 Dense 레이어 각각에서 과적합을 막기 위해서"
<요약>
Dense 레이어마다 복잡도가 높아지니까
과적합 막으려고 Dropout을 각 레이어 뒤에 따로 넣은 것
(특히 0.3은 꽤 강하게 떨어뜨리는 편)

2. Softmax는 어떤 원리로 클래스 하나를 선택?
👉 출력값(로짓)을 확률처럼 변환해서, 가장 높은 확률을 가진 클래스를 선택
디테일하게 보면:
마지막 Dense 층은 각 클래스에 대한 "로짓 (logit)"을 출력 (그냥 점수 같은 값)
Softmax는 이 로짓들을 "확률 분포"처럼 변환
수식을 통해 계산 후 np.argmax를 써서 가장 높은 확률을 가진 클래스를 최종 예측값으로 선택

3. EarlyStopping은 어떻게 학습을 멈추는 기준을 잡는걸까?
👉 validation 데이터에 대해 더 이상 성능이 좋아지지 않으면 학습을 멈추는 기술이야.

구체적으로:
monitor='val_loss': validation set에 대한 loss를 지켜봄
patience=5: 만약 5번 연속 validation loss가 개선되지 않으면 멈춤
즉, "validation loss가 5번 에폭 동안 계속 줄지 않으면 → 과적합이 시작됐다고 판단 → 학습 중단"
추가로, restore_best_weights=True 옵션 덕분에, 성능이 가장 좋았던 시점의 모델 가중치로 자동으로 복원
"""
"""
Weighted F1 Score: 0.9901
Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       663
           1       0.99      0.98      0.98       419
           2       1.00      0.99      1.00      2186
           3       0.86      0.63      0.73        19
           4       0.99      0.99      0.99       668
           5       0.93      0.93      0.93        45

    accuracy                           0.99      4000
   macro avg       0.96      0.92      0.94      4000
weighted avg       0.99      0.99      0.99      4000

1. 전체 정확도 (Accuracy)
99% (0.99)
→ 4000개 샘플 중 거의 다 맞췄다는 뜻이야.

2. 클래스별 성능

클래스	precision	recall	f1-score	샘플 개수 (support)	설명
0	0.98	1.00	0.99	663개	거의 완벽하게 분류
1	0.99	0.98	0.98	419개	아주 잘 분류
2	1.00	0.99	1.00	2186개	사실상 완벽
3	0.86	0.63	0.73	19개	성능 떨어짐 ⚡
4	0.99	0.99	0.99	668개	매우 좋음
5	0.93	0.93	0.93	45개	괜찮음
3. 특별히 주의할 부분
클래스 3 (support가 19개밖에 안 되는 소수 클래스)은 recall이 0.63으로 낮아.
→ 이 말은 실제 3인 데이터 중 63%만 맞추고, 37%는 놓쳤다는 뜻이야.
→ 데이터 수가 적거나, 모델이 이 클래스를 잘 학습 못했을 가능성이 있어.

4. 평균 성능
macro avg (모든 클래스를 동등하게 본 평균)

precision: 0.96

recall: 0.92

f1-score: 0.94

weighted avg (각 클래스 비율을 고려한 평균)

모두 0.99로 아주 높아.

✨ 총평
모델은 전반적으로 매우 뛰어난 성능을 보이고 있어!

단, 클래스 3 개선을 고민해보는 게 좋아 보여.

클래스 3 데이터 수를 늘리거나

클래스 3에 가중치를 더 주는 방식 (class_weight="balanced" 같은 설정) 고려 가능.
"""
# 6-1. 딥러닝 모델 만들기
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # 클래스 개수 만큼
])

nn_model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

# 6-2. 딥러닝 학습
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 6-3. 딥러닝 평가
nn_preds = np.argmax(nn_model.predict(X_test), axis=1)

print("Neural Network Classification Report")
print(classification_report(y_test, nn_preds, target_names=le_body.classes_))

# 1. 모델 예측
y_pred_probs = nn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # 확률 -> 클래스 선택
y_true_classes = y_test  # 그냥 y_test 사용

# 2. Confusion Matrix 계산
cm = confusion_matrix(y_true_classes, y_pred_classes)

# 3. Confusion Matrix 시각화
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_body.classes_, yticklabels=le_body.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 4. Classification Report 출력
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=le_body.classes_))

# 5. F1 Score 출력
from sklearn.metrics import f1_score

f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")

# 6-4. 딥러닝 혼동행렬 시각화
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, nn_preds), annot=True, fmt='d', cmap='Greens',
            xticklabels=le_body.classes_, yticklabels=le_body.classes_)
plt.title('Neural Network Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 6-5. 학습 과정 시각화 (Loss, Accuracy)
plt.figure(figsize=(14,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. 모델 예측
y_pred_probs = nn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # 확률 -> 클래스 선택
y_true_classes = y_test  # 여기! 그냥 y_test를 그대로 써

# 2. Confusion Matrix 계산
cm = confusion_matrix(y_true_classes, y_pred_classes)

# 3. Confusion Matrix 시각화
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 4. Classification Report 출력
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))
