# Spring의 Application.java에 해당
from fastapi import FastAPI
from app.api.v1 import routes

app = FastAPI()

# API 엔드포인트 등록
app.include_router(routes.router)
