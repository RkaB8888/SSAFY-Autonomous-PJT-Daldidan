# FastAPI 앱 시작점 (AI 추론 서버)
from fastapi import FastAPI
from api.v1 import routes

app = FastAPI(title="Apple Sugar Prediction API")

# /api/v1/predict 엔드포인트 등록
app.include_router(routes.router)
