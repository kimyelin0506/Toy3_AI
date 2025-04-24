
#  프로젝트명: 신사숙녀 - 개인 맞춤형 상품 추천 시스템

##  프로젝트 개요
"유행"보다 "개인의 개성과 취향"이 중요한 시대.  
신사숙녀 프로젝트는 AI 기반의 개인 맞춤형 상품 추천 시스템을 구축하여, 고객의 신체 특성과 후기 감성을 반영한 추천을 가능하게 합니다.

##  개발 방식 및 데이터 설명

1. **기존 이커머스 웹사이트에서 필요한 서비스 정의**
2. **데이터 수집**: Kaggle, 공공데이터포털 등
3. **동일한 상품 기반 데이터 전처리**
4. **개별 모델 학습 및 튜닝**
5. **신뢰성 있는 2가지 모델을 선정해 분석 및 시각화 수행**

---

##  머신러닝 이론

- **K-Means**
  - 비지도 학습 알고리즘
  - 유클리디안 거리 기반 클러스터링
  - Elbow 기법: 적정 K 찾기
  - Silhouette Score: 군집 응집력/분리도 평가

---

##  개인 맞춤 추천 모델 [K-Means 기반]

###  사용된 Feature
- **수치형**: `bmi`, `size`, `age`
- **범주형**: `fit`, `body type` (→ 원핫 인코딩)

###  전처리
- **결측치 처리**:
  - 수치형 → 평균값 대체
  - 범주형 → 최빈값 대체
- **범주형 가중치 처리**:
  - (원핫 인코딩된 컬럼 수) / (전체 컬럼 수)
- **스케일링**: MinMaxScaler

###  KMeans 결과 (1)
- 최적의 k = 7
- Silhouette Score: 0.665
- **Elbow Plot 시각화**
- **t-SNE 시각화**: 클러스터 경계 뚜렷, body type 중심으로 나뉨
- **클러스터 평균값 시각화**: 체형 특징별 맞춤 추천 가능

---

###  KMeans 결과 (2) - 대용량 데이터 기준
- 데이터 구조: 20~30대 여성 타겟, 2만명 / 15만개 샘플
- 사용 특성: `weight`, `height`, `size`, `age`, `body type`
- 추천 기준: `fit = fit`, `rating ≥ 8점`
- **K=6에서 최적화**, Silhouette Score: 0.3~0.4

####  PCA 시각화
- PC1: 체형 크기 (size, weight, height)
- PC2: 나이 그룹
- → 체형별 3층 구조, 각 층 내 다양한 나이대 분포

####  t-SNE 시각화 및 군집 설명
- Cluster 0: 모래시계/운동형/25-29세, 평균 사이즈 13.84
- Cluster 1: 작은체형/30-34세, 평균 사이즈 3.84
- Cluster 2: 모래시계/운동형/35-39세, 평균 사이즈 10.38
- Cluster 3: 모래시계/풀버스트, 평균 사이즈 21.67
- Cluster 4: 작은체형/25-29세, 평균 사이즈 4.60
- Cluster 5: 모래시계/배형/30-34세, 평균 사이즈 12.67

---

##  신체 치수를 이용한 체형 분류

###  알고리즘 비교

| 알고리즘 | 선택 이유 | 주요 결과 |
|----------|-----------|-----------|
| RandomForest | 다양한 특성 조합 학습 가능, 과대적합 방지 | 정밀도/재현율 높음, 소수 클래스 예측 어려움 |
| K-Means | 군집 중심 체형 분석에 직관적 | Silhouette: 0.158~0.293, Accuracy: 62~69% |
| MLP | 복잡한 피처 간 비선형 관계 학습 | Accuracy: 99%, F1-score 높음, 소수 클래스 보완 필요 |

###  데이터 전처리
- 바디타입 7종 정의
- `height`와 `weight` 기반 파생 feature 생성
- 타겟 인코딩 및 전처리 완료

---

##  후기 감정 분석 및 실제 점수 비교 [DistilBERT]

###  사용 배경
- DistilBERT: 경량화된 BERT 모델, 사전학습 기반, 감정 분석에 적합

###  학습 및 결과
- 리뷰 20만 건 분석
- 감정(긍정/부정)과 별점 일치율 ≈ **85.47%**
- 기준:
  - 긍정 감정 → 별점 ≥ 8
  - 부정 감정 → 별점 < 8
- 인사이트:
  - 리뷰 감정 기반 별점 예측은 신뢰성 있음
  - 실제 추천 기준으로 "8점 이상 리뷰 상품" 전략 타당

---

##  프로젝트 회고

- A: "코드와 데이터보다 팀워크가 완성의 핵심이었다."
- B: 실제 서비스를 구현하는 경험이 유익했다.
- C: 실습을 통해 AI를 가까이 접할 수 있었다.
- D: 협업으로 AI 학습이 수월해졌다.
- E: 머신러닝을 다시 되짚고 딥하게 배울 필요를 느꼈다.

---

##  향후 개발 방향

- **FastAPI 서버 연동 → 기존 쇼핑몰에 적용**
- **LLM 기반 RAG 챗봇 서비스 기능 추가**
- **구매 이력, 스타일 선호도 기반 추천 기능 확대**

---

##  참고 문헌

- [계층적 군집분석을 활용한 아시아 컨테이너 항만 클러스터링 측정](http://kportea.or.kr/filedown/Treatise/2022/3.%20%5B%ED%95%AD%EB%A7%8C%EA%B2%BD%EC%A0%9C%2037%EA%B6%8C%201%ED%98%B8%5D%20%EA%B3%84%EC%B8%B5%EC%A0%81%20%EA%B5%B0%EC%A7%91%EB%B6%84%EC%84%9D(%EC%B5%9C%EB%8B%A8,%20%EC%B5%9C%EC%9E%A5,%20%ED%8F%89%EA%B7%A0,%20%EC%A4%91%EC%95%99%EC%97%B0%EA%B2%B0)%EB%B0%A9%EB%B2%95%EC%97%90%20%EC%9D%98%ED%95%9C%20%EC%95%84%EC%8B%9C%EC%95%84%20%EC%BB%A8%ED%85%8C%EC%9D%B4%EB%84%88%20%ED%95%AD%EB%A7%8C%EC%9D%98%20%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0%EB%A7%81%20%EC%B8%A1%EC%A0%95%20%EB%B0%8F%20%EC%8B%A4_.pdf)
- [Body shape classification using measurements (Springer, 2018)](https://link.springer.com/chapter/10.1007/978-3-319-77700-9_8)
- [Human body shape prediction and classification for mass customization (Elsevier, 2019)](https://biomedicaloptics.spiedigitallibrary.org/conference-proceedings-of-spie/4309/0000/3D-measurement-of-human-body-for-apparel-mass-customization/10.1117/12.410883.short)
