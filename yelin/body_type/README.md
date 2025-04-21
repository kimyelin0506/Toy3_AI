| 항목         | 1번(Random Forest)                   | 2번(Neural Network)                | 3번(K-Means Clustering)            |
|--------------|---------------------------------------|-------------------------------------|------------------------------------|
| 기법         | 지도학습 (Random Forest)              | 지도학습 (딥러닝 MLP)              | 비지도학습 (K-Means)               |
| 목표         | 체형(body type) 분류                  | 체형(body type) 분류                | 체형 기반 군집화 및 대표 체형 찾기 |
| 입력 데이터  | 전처리된 피처(X_train)                 | 전처리된 피처(X_train)             | 원본 신체 치수(Bust, Waist, Hips, Height 등) |
| 타겟(label)  | body_type (정답 존재)                 | body_type (정답 존재)               | 없음 (label 없이 스스로 군집)      |
| 모델 종류    | Random Forest Classifier              | Multi-layer Perceptron (Dense 층 3개) | K-Means Clustering (n_clusters=5) |
| 평가 방법    | Accuracy, F1 Score, Confusion Matrix  | Accuracy, F1 Score, Confusion Matrix | 없음 (군집 해석)                  |
| 특징         | - 빠르고 튼튼<br>- 과적합 잘 버팀     | - 복잡한 패턴 잘 학습<br>- 약간 오버핏 주의 | - 분류 없이 그룹화<br>- 대표 체형 분석 가능 |
| 시각화       | Confusion Matrix (Blues 컬러)         | Confusion Matrix + 학습 그래프(loss/acc) | 없음(결과 출력)                  |
| 특이점       | 트리 기반 앙상블                      | 드롭아웃으로 과적합 방지             | 군집별로 '대표 체형' 지정           |


1번: Random Forest
이미 정답(y) 이 있는 상황에서 학습.
여러 개 Decision Tree를 조합해서 예측.
결과 해석: confusion matrix로 클래스별 정확도 확인.

⭐️ 장점: 빠르고 튼튼, 과적합 적음.
⭐️ 단점: 너무 많은 트리로 모델이 무거워질 수 있음.

2번: Neural Network (MLP)
마찬가지로 정답(y) 이 있는 상태에서 학습.
Dense 층(128 → 64 → Output) + Dropout 두 번.
softmax로 다중 클래스 분류.
EarlyStopping 적용해서 과적합 방지.
학습 과정(loss, acc) 시각화까지 포함.

⭐️ 장점: 복잡한 데이터 패턴 잘 잡음.
⭐️ 단점: 튜닝(레이어 수, 드롭아웃 비율 등) 필요, Overfitting 주의.

3번: K-Means Clustering
정답(y) 없음.
신체 치수만 가지고 사람들을 그룹화.
이후, 개인별 체형 분류 + 군집별 대표 체형 계산.
목적은 "비슷한 사람들끼리 그룹핑하고", "그룹 대표 체형 파악"하는 것.

⭐️ 장점: 분류 없이 데이터 패턴을 스스로 발견.
⭐️ 단점: n_clusters 수를 직접 정해야 함. (5로 고정했음)