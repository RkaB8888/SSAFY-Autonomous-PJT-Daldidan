# Spring의 Application.java에 해당
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from fastapi_socketio import SocketManager
from api.v1 import routes
# import random

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # 핵심: mount_location 명시
# socket_manager = SocketManager(app=app, async_mode="asgi", mount_location="/socket.io")


# @socket_manager.on("connect")
# async def on_connect(sid, environ):
#     print(f"✅ 클라이언트 연결됨: {sid}")


# @socket_manager.on("frame")
# async def on_frame(sid, data):
#     print(f"📷 프레임 수신: {str(data)[:30]}...")

#     count = random.randint(1, 4)
#     results = []

#     max_x = 360 - 100
#     max_y = 640 - 100

#     for i in range(count):
#         brix = round(random.uniform(5.0, 15.0), 1)
#         box = [random.randint(0, max_x), random.randint(0, max_y), 100, 100]
#         results.append({"id": i + 1, "brix": brix, "box": box})

#     await socket_manager.emit("prediction", {"results": results}, to=sid)


# @socket_manager.on("disconnect")
# async def on_disconnect(sid):
#     print(f"❌ 연결 해제: {sid}")


# API 엔드포인트 등록
app.include_router(routes.router)
