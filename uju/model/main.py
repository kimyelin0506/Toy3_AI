# 상춤 추천 서비스(기존 의류 쇼핑몰과 연동됨)
# uvicorn main:app --reload (FastAPI 실행)
# 상품 추천서비스 spring boot rest API 실행해야 홈페이지에서 보임

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# FastAPI 애플리케이션 초기화
app = FastAPI()

# 사전 학습된 모델과 전처리 도구들 불러오기
model_bundle = joblib.load("optimal_kmeans_k6_model_full2.pkl")
kmeans = model_bundle['kmeans']              # KMeans 모델
scaler = model_bundle['scaler']              # 수치형 특성 스케일링 도구
encoder_age = model_bundle['encoder_age']    # 연령대 원-핫 인코더
encoder_body = model_bundle['encoder_body']  # 체형 원-핫 인코더

# 사용자 입력 형식을 정의 (요청 바디 모델)
class UserRequest(BaseModel):
    height: float
    weight: float
    size: float
    age: int
    body_type: str

# 응답 형식 정의
class RecommendationResponse(BaseModel):
    cluster: int

# 추천 API 엔드포인트 정의
@app.post("/recommend", response_model=RecommendationResponse)
def recommend(user: UserRequest):
    try:
        # 사용자 입력을 DataFrame으로 변환
        user_df = pd.DataFrame([{
            "height_cm": user.height,
            "weight_kg": user.weight,
            "size": user.size,
            "age": user.age,
            "body type": user.body_type
        }])

        # 연령대 그룹화 (범주형 변수로 변환)
        user_df["age_group"] = pd.cut(
            user_df["age"],
            bins=[19, 25, 30, 35, 40],
            labels=["20-24", "25-29", "30-34", "35-39"]
        )

        # 수치형 변수 스케일링
        num_features = scaler.transform(user_df[["height_cm", "weight_kg", "size"]])

        # 연령대 인코딩 후 가중 평균 처리
        age_encoded = encoder_age.transform(user_df[["age_group"]])
        age_weights = np.ones(age_encoded.shape[1]) / age_encoded.shape[1]
        age_encoded = age_encoded * age_weights

        # 체형 인코딩 후 가중 평균 처리
        body_encoded = encoder_body.transform(user_df[["body type"]])
        body_weights = np.ones(body_encoded.shape[1]) / body_encoded.shape[1]
        body_encoded = body_encoded * body_weights

        # 모든 특성 결합 (수치형 + 인코딩된 연령대 + 인코딩된 체형)
        features = np.hstack([num_features, age_encoded, body_encoded])

        # KMeans로 클러스터 예측
        cluster = int(kmeans.predict(features)[0])

        # 클러스터 번호 반환
        return RecommendationResponse(cluster=cluster)

    # 예외 발생 시 500 에러와 함께 메시지 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
