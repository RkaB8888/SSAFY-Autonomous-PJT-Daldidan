# 📚 BE (Backend) 서버 구조 설명

---

## 📌 프로젝트 개요

본 프로젝트는 FastAPI 기반의 **게이트웨이 서버**로,  
- 프론트엔드와 **WebSocket(socket.io)** 통신  
- 로컬 AI 추론 서버(`ai/`)와의 HTTP API 연동  
을 담당합니다.

AI 모델 자체는 **로컬의 `ai/` 서버에서 추론**하며, 이 `be/` 서버는 **중계 역할만 수행**합니다.

---

## 📁 디렉토리 구조

```text
BE/
├── .gitignore
├── README.md
├── requirements.txt
│
├── be/                                ← EC2에서 실행될 WebSocket + API 서버
│   ├── main.py                        # FastAPI + socket.io 진입점
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── routes.py              # REST API 라우터 정의
│   │       └── __init__.py
│   │
│   ├── core/
│   │   └── config.py                  # 설정 및 환경변수 관리
│   │
│   ├── schemas/
│   │   ├── api.py                     # REST API용 요청/응답 스키마
│   │   ├── socket.py                  # WebSocket용 스키마
│   │   └── __init__.py
│   │
│   ├── services/
│   │   └── remote_model.py           # 로컬 AI 서버와 HTTP 통신
│   │
│   ├── socketio_app/
│   │   ├── events.py                 # socket.io 이벤트 처리
│   │   ├── manager.py                # 연결 관리 유틸
│   │   └── __init__.py
│   │
│   └── utils/
│       └── image.py                  # 이미지 전처리 유틸
│
├── ai/                                ← 로컬에서 실행될 AI 추론 서버
│   ├── main.py                        # FastAPI 진입점
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── routes.py             # /predict 요청 처리
│   │       └── __init__.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── schemas/
│   │   └── predict.py                # 모델 추론 요청/응답 스키마
│   │
│   ├── services/
│   │   ├── predict_service.py        # Dispatcher
│   │   ├── model_a/
│   │   │   ├── loader.py             # model_a 로딩
│   │   │   └── predictor.py          # model_a 추론
│   │   └── model_b/
│   │       ├── loader.py
│   │       └── predictor.py
│   │
│   ├── models/
│   │   ├── model_a/
│   │   │   └── model.pt              # 가중치 파일 (예시)
│   │   └── model_b/
│   │       └── model.pt
│   │
│   └── utils/
│       └── image.py
```

---

## 📌 BE(be/) 디렉토리 설명

| 경로 | 설명 |
|------|------|
| `be/main.py` | FastAPI + socket.io 서버 실행 진입점 |
| `api/v1/routes.py` | REST API 라우터 등록 |
| `socketio_app/events.py` | WebSocket 이벤트 처리 (`image_data`, `connect`, `disconnect`) |
| `services/remote_model.py` | 로컬 AI 서버와의 HTTP 통신 (예: POST `/predict`) |
| `schemas/socket.py` | WebSocket 통신용 데이터 모델 |
| `schemas/api.py` | REST API 요청/응답용 스키마 |
| `utils/image.py` | 이미지 base64 디코딩 및 전처리 유틸 |

---

## 📌 AI(ai/) 디렉토리 설명

| 경로 | 설명 |
|------|------|
| `ai/main.py` | AI 추론 서버의 FastAPI 실행 진입점 |
| `services/predict_service.py` | 모델별 분기(Dispatcher) 로직 |
| `services/model_a/` | 모델 A의 로딩 및 예측 |
| `services/model_b/` | 모델 B의 로딩 및 예측 |
| `models/` | 각 모델의 가중치 파일 저장 위치 |
| `schemas/predict.py` | 추론 요청/응답 스키마 정의 |
| `utils/image.py` | 이미지 디코딩/전처리 유틸 |

---

## 📌 모델 가중치 Git 무시 설정

`.gitignore`에는 다음과 같은 규칙이 포함되어 있어 모델 가중치 파일이 Git에 업로드되지 않습니다:

```gitignore
ai/models/*/*.pth
ai/models/*/*.h5
ai/models/*/*.pt
!ai/models/**/.gitkeep
```

---

## 📌 AI 서버 헬스 체크

- `GET /api/v1/health` 요청을 통해 AI 서버가 정상적으로 실행 중인지 확인할 수 있습니다.

```json
{
  "status": "AI server is running"
}
```

---

## 📌 AI 서버 예측 요청 API

**POST /api/v1/predict**

| 필드 | 타입 | 설명 |
|------|------|------|
| `model_name` | string | 사용할 모델 이름 (`model_a`, `model_b`) |
| `image_base64` | string | base64로 인코딩된 이미지 데이터 |

**예시 요청:**
```json
{
  "model_name": "model_a",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**예시 응답:**
```json
{
  "label": "Apple",
  "confidence": 0.92
}
```

---

## 📌 WebSocket 구조 개요

- `be/socketio_app/` 내부에서 socket.io 기반 WebSocket 이벤트를 처리합니다.
- 클라이언트로부터 받은 이미지 데이터를 로컬 AI 서버로 전달하고, 예측 결과를 다시 클라이언트에 전송합니다.
- 예시 이벤트:
  - `image_data`: 이미지 수신
  - `prediction_result`: 예측 결과 전송

---

## 📌 향후 확장 계획

| 기능 | 설명 |
|------|------|
| 다양한 모델 실험 | Dispatcher 구조로 여러 AI 모델 비교 실험 지원 |
| WebSocket 고도화 | 이미지 버퍼링/에러처리 강화 |
| API 통합 테스트 | WebSocket ↔ API ↔ AI 모델 전과정 테스트 및 자동화 예정 |

---

## ✅ 요약

> 이 프로젝트는 WebSocket 기반 사용자 인터페이스와 AI 모델 실험을 분리하여 설계하였으며,  
> **유연성 높은 Dispatcher 기반 구조와 socket.io 기반 실시간 통신**을 통해 실험과 서비스 양쪽 모두 대응이 가능하도록 구성되어 있습니다.
