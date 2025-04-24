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

랜덤포레스트(Random Forest) 쪽은
→ 미리 "정답" 레이블(body_type_apple, body_type_athletic, body_type_full bust, body_type_hourglass, body_type_pear, body_type_straight & narrow)이 존재해.
→ 즉, classify_body_type() 함수로 이미 "개인 체형(body type)"을 딱 정해놓은 다음, 그걸 타겟(y값)으로 학습하는 거야. → 그래서 6개 클래스가 고정된 상태로 모델을 돌리는 거지.

K-Means 쪽은
→ 정답(타겟 레이블)이 없는 상태야.
→ 데이터 포인트들의 (Waist, Hips, Bust/Chest, BMI) 이런 특징(feature)만 보고, K-Means가 "알아서" 비슷한 것끼리 7개로 나눈다는 거야 (n_clusters=7 설정).
→ 그래서 실제로는 "body_type_apple" 이런 태그는 모른 채, 그냥 군집 번호(0~6번)만 부여한 거고,
→ 군집마다 "대표 체형"을 나중에 붙이는 과정이 따로 들어간 거야. (군집 안에서 제일 많이 나온 개인 체형을 대표로!)

그러니까 질문한 거처럼:
KMeans 군집 중에서 어떤 클래스에 소속된 데이터가 하나도 없는 경우가 있을 수도 있고,
또 원래 개인 체형 6개 중 일부는 특정 군집에는 하나도 없을 수도 있어.
(특히 n_clusters=7로 6개보다 군집 수를 더 많이 잡았으니까 더 그럴 확률이 높아.)

핵심 요약
RandomForest는: 이미 타겟이 "명확하게 6개" 존재.
KMeans는: 타겟 없이 "알아서 7개" 군집으로 묶음. → 원래 존재하는 body type 수랑은 직접적인 관계 없음.
그래서 "어떤 군집에는 특정 body type이 하나도 없을 수도 있다" 가 맞아!

| 모델           | 전체 정확도     | 소수 클래스 처리      | 특징                        | 주의점                     |
|----------------|----------------|------------------------|-----------------------------|----------------------------|
| Random Forest  | ⭐ 97%          | ❌ 낮은 성능           | 이상치에 강함               | imbalance 보완 필요        |
| K-Means        | 👍 약 70% 일치  | -                      | full bust 군집화 매우 우수 | 비지도, 라벨 없음          |
| MLP            | 🌟 99%          | ⚠ 일부 클래스 낮음     | 복잡한 관계 학습 가능      | 이상치 민감                |
