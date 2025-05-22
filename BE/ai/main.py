from fastapi import FastAPI, Request
from api.v1 import routes
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import services.yolov8


class UnlimitedSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        return await call_next(request)


app = FastAPI(title="Apple Sugar Prediction API")
app.add_middleware(UnlimitedSizeMiddleware)

# /api/v1/predict 엔드포인트 등록
app.include_router(routes.router)
