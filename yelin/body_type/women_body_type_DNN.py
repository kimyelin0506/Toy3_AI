# 1. 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from pre_women_body_type_data import X_train, X_test, y_train, y_test, y_encoded, le_body
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
