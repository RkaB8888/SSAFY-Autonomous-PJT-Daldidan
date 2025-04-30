# 여기는 Spring Boot의 Controller에 해당
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "OK"}
