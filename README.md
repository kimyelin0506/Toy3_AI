
#  프로젝트명: 신사숙녀W - AI 기반 개인 맞춤형 상품 추천 시스템

##  프로젝트 개요
- "유행"보다 "개인의 개성과 취향"이 중요한 시대.
- 신사숙녀W 프로젝트는 AI 기반의 2030 여성 고객 맞춤형 상품 추천 시스템을 구축하여, 고객의 신체 특성과 후기 감성을 반영한 추천 서비스를 제공합니다.

##  개발 방식 및 데이터 설명
1. **기존 이커머스 웹사이트에서 필요한 서비스 정의**
2. **데이터 수집**: Kaggle - 여성 의류 쇼핑몰 Rent The Runway(상품 데이터), woman body syn(신체 사이즈 데이터)
3. **데이터 전처리**
4. **모델 학습 및 튜닝**
5. **모델 분석 및 시각화 수행**

---

##  1. 개인 맞춤 추천 모델 [K-Means 기반]

###   [K-Means 결과 (1)](./lbw/shoppingMallCluster.ipynb)
- 사용 특성: 
  - **수치형**: `bmi`, `size`, `age`
  - **범주형**: `fit`, `body type` (→ 원핫 인코딩)
- 전처리
  - **결측치 처리**:
    - 수치형 → 평균값 대체
    - 범주형 → 최빈값 대체
  - **범주형 가중치 처리**:
    - (원핫 인코딩된 컬럼 수) / (전체 컬럼 수)
  - **스케일링**: MinMaxScaler
- 최적의 k = 7
- Silhouette Score: 0.665
- **Elbow Plot 시각화**
- **t-SNE 시각화**: 클러스터 경계 뚜렷, body type 중심으로 나뉨
- **클러스터 평균값 시각화**: 체형 특징별 맞춤 추천 가능
- [상세 정보 README.md 확인](./lbw/README.md)
---

###   [K-Means 결과 (2) - 대용량 데이터 기준](./uju/model/4_Feature_Engineering_full.ipynb)
- 데이터 구조: 20~30대 여성 타겟, 2만개 샘플 / 15만개 전체
- 사용 특성: `weight`, `height`, `size`, `age`, `body type`
- - 전처리
  - **결측치 처리**:
    - 수치형 → 평균값 대체(몸무게는 사이즈별 평균값; 몸무게와 사이즈 상관관계: 0.855)
    - 범주형 → 최빈값 대체
  - **범주형 가중치 처리**:
    - (각 범주형 피처-도메인 개수만큼 나누기, 합은 1)
  - **스케일링**: MinMaxScaler
- 추천 기준: `fit = fit`, `rating ≥ 8점`
- **K=6에서 최적화**, Silhouette Score: 0.3~0.4
- PCA 시각화 - 전역 구조
  - PC1: 체형 크기 (size, weight, height)
  - PC2: 나이 그룹
  - → 체형별 3층 구조, 각 층 내 다양한 나이대 분포
- t-SNE 시각화 및 군집 설명 - 국소 구조
  - Cluster 0: 모래시계/운동형/25-29세, 평균 사이즈 13.84
  - Cluster 1: 작은체형/30-34세, 평균 사이즈 3.84
  - Cluster 2: 모래시계/운동형/35-39세, 평균 사이즈 10.38
  - Cluster 3: 모래시계/풀버스트, 평균 사이즈 21.67
  - Cluster 4: 작은체형/25-29세, 평균 사이즈 4.60
  - Cluster 5: 모래시계/배형/30-34세, 평균 사이즈 12.67
- [상세 정보 README.md 확인](./uju/model/README.md)

---

##  2. 신체 치수를 이용한 체형 분류 [Random Forest, Neural Network, K-Means]
- [전처리](./yelin/body_type/pre_women_body_type.py)
- [상세 정보 README.md 확인](./yelin/body_type/info/README.md)

### 사용한 알고리즘 비교
| 항목         | [1번(Random Forest)](./yelin/body_type/RandomForestClassifier_women_body_type.py) | [2번(Neural Network)](./yelin/body_type/MLP_women_body_type.py) | [3번(K-Means Clustering)](./yelin/body_type/Kmeans_women_body_type.py)           |
|--------------|----------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------|
| 기법         | 지도학습 (Random Forest)                                                             | 지도학습 (딥러닝 MLP)                                                 | 비지도학습 (K-Means)               |
| 목표         | 체형(body type) 분류                                                                 | 체형(body type) 분류                                               | 체형 기반 군집화 및 대표 체형 찾기 |
| 입력 데이터  | 전처리된 피처(X_train)                                                                 | 전처리된 피처(X_train)                                               | 원본 신체 치수(Bust, Waist, Hips, Height 등) |
| 타겟(label)  | body_type (정답 존재)                                                                | body_type (정답 존재)                                              | 없음 (label 없이 스스로 군집)      |
| 모델 종류    | Random Forest Classifier                                                         | Multi-layer Perceptron (Dense 층 3개)                            | K-Means Clustering (n_clusters=5) |
| 평가 방법    | Accuracy, F1 Score, Confusion Matrix                                             | Accuracy, F1 Score, Confusion Matrix                           | 없음 (군집 해석)                  |
| 특징         | - 빠르고 튼튼<br>- 과적합 잘 버팀                                                           | - 복잡한 패턴 잘 학습<br>- 약간 오버핏 주의                                   | - 분류 없이 그룹화<br>- 대표 체형 분석 가능 |
| 시각화       | Confusion Matrix (Blues 컬러)                                                      | Confusion Matrix + 학습 그래프(loss/acc)                            | 없음(결과 출력)                  |
| 특이점       | 트리 기반 앙상블                                                                        | 드롭아웃으로 과적합 방지                                                  | 군집별로 '대표 체형' 지정           |

###  알고리즘 결과 비교

| 알고리즘 | 선택 이유 | 주요 결과 |
|----------|-----------|-----------|
| RandomForest | 다양한 특성 조합 학습 가능, 과대적합 방지 | 정밀도/재현율 높음, 소수 클래스 예측 어려움 |
| K-Means | 군집 중심 체형 분석에 직관적 | Silhouette: 0.158~0.293, Accuracy: 62~69% |
| MLP | 복잡한 피처 간 비선형 관계 학습 | Accuracy: 99%, F1-score 높음, 소수 클래스 보완 필요 |

##  3. 후기 감정 분석 및 실제 점수 비교 [DistilBERT]

###  사용 배경
- DistilBERT: 경량화된 BERT 모델, 사전학습 기반, 감정 분석에 적합

###  학습 및 결과
- 리뷰 20만 건 분석
- [학습 모델](./yelin/review/DistilBERT_text_model.py)
- 감정(긍정/부정)과 별점 일치율 ≈ **85.47%**
- [오차 확인](./yelin/review/review_rating.py)
- 기준:
  - 긍정 감정 → 별점 ≥ 8
  - 부정 감정 → 별점 < 8
- 인사이트:
  - 리뷰 감정 기반 별점 예측은 신뢰성 있음
  - 실제 추천 기준으로 "8점 이상 리뷰 상품" 전략 타당
- [상세 정보 README.md 확인](./yelin/review/info/README.md)

---

## 향후 개발 방향

- **FastAPI 서버 연동 및 기존 쇼핑몰 시스템 통합**
  - 사용자 입력(신체 치수 등)을 기반으로 바디 타입 예측 기능 제공  
  - 사용자 맞춤 상품 추천 시스템 구축  
  - 리뷰 품질 관리 및 신뢰도 낮은 리뷰 필터링 기능 추가  

- **추천 시스템 고도화**
  - 기존 체형 기반 추천 외에 **구매 이력**, **스타일 선호도** 등 다양한 사용자 데이터를 수집  
  - 더 정교하고 개인화된 추천 로직 개발  
  - 사용자의 선호 패턴을 학습하여 실시간 추천 제공  

- **LLM + RAG 기반 챗봇 도입**
  - 상품 관련 문의, 체형에 맞는 스타일 상담 등 **대화형 쇼핑 지원**  
  - 자체 데이터(RAG 기반)를 활용하여 신뢰도 높은 응답 제공  
  - 사용자 경험(UX) 향상 및 고객 만족도 증대 목표  

---

##  참고 문헌

- [계층적 군집분석을 활용한 아시아 컨테이너 항만 클러스터링 측정](http://kportea.or.kr/filedown/Treatise/2022/3.%20%5B%ED%95%AD%EB%A7%8C%EA%B2%BD%EC%A0%9C%2037%EA%B6%8C%201%ED%98%B8%5D%20%EA%B3%84%EC%B8%B5%EC%A0%81%20%EA%B5%B0%EC%A7%91%EB%B6%84%EC%84%9D(%EC%B5%9C%EB%8B%A8,%20%EC%B5%9C%EC%9E%A5,%20%ED%8F%89%EA%B7%A0,%20%EC%A4%91%EC%95%99%EC%97%B0%EA%B2%B0)%EB%B0%A9%EB%B2%95%EC%97%90%20%EC%9D%98%ED%95%9C%20%EC%95%84%EC%8B%9C%EC%95%84%20%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88%20%ED%95%AD%EB%A7%8C%EC%9D%98%20%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81%20%EC%B8%A1%EC%A0%95%20%EB%B0%8F%20%EC%8B%A4_.pdf)
- [Body shape classification using measurements (Springer, 2018)](https://link.springer.com/chapter/10.1007/978-3-319-77700-9_8)
- [Human body shape prediction and classification for mass customization (Elsevier, 2019)](https://biomedicaloptics.spiedigitallibrary.org/conference-proceedings-of-spie/4309/0000/3D-measurement-of-human-body-for-apparel-mass-customization/10.1117/12.410883.short)
