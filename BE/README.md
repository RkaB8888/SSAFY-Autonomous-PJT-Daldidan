# 📚 BE (Backend) 프로젝트 구조 설명

---

# 📌 BE (Backend) 프로젝트 구조

본 프로젝트는 **FastAPI**를 기반으로 구성된 백엔드 서버입니다.  
주요 역할은 **AI 모델 서비스 제공** 및 **WebSocket 기반 이미지 프레임 수신 및 처리**입니다.

---

# 📁 디렉토리 구조

```text
BE/
├── .gitignore              # Git 무시 파일 설정
├── README.md               # 프로젝트 설명
├── requirements.txt        # Python 의존성 패키지 목록
│
├── app/
│   ├── main.py              # FastAPI 앱 시작점
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── routes.py    # 일반 HTTP API 라우터
│   │       └── ws_routes.py # WebSocket API 라우터
│   │
│   ├── core/
│   │   └── config.py        # 전역 설정 관리
│   │
│   ├── models/
│   │   ├── model_a/         # model_a 가중치(.pth 등) 저장
│   │   └── model_b/         # model_b 가중치(.pth 등) 저장
│   │
│   ├── schemas/
│   │   └── schemas.py       # API 요청/응답 데이터 스키마 정의 (Pydantic 사용)
│   │
│   ├── services/
│   │   ├── predict_service.py # 모델별 분기 처리(Dispatcher)
│   │   ├── model_a/
│   │   │    ├── loader.py   # model_a 모델 로딩
│   │   │    └── predictor.py# model_a 예측 처리
│   │   ├── model_b/
│   │   │    ├── loader.py
│   │   │    └── predictor.py
│   │   └── ws/
│   │       └── connection_manager.py # WebSocket 연결 및 메시지 관리
│   │
│   └── utils/
│       └── image_preprocess.py # 공통 유틸 함수 (ex. 이미지 전처리)
```

---

# 📌 디렉토리/파일 설명

| 경로 | 설명 |
|:---|:---|
| `app/main.py` | FastAPI 서버 시작 스크립트 |
| `api/v1/routes.py` | 일반 HTTP API 라우터 등록 |
| `api/v1/ws_routes.py` | WebSocket API 라우터 등록 (이미지 프레임 수신) |
| `core/config.py` | 서버 설정 및 환경 변수 관리 |
| `models/` | AI 모델 가중치(.pth, .h5 등) 파일 저장 위치 |
| `schemas/schemas.py` | 요청/응답 데이터 스키마 정의 (Pydantic) |
| `services/` | 비즈니스 로직 처리 (모델 로딩, 예측, WebSocket 연결 관리) |
| `utils/image_preprocess.py` | 이미지 전처리 유틸리티 함수 |

---

# 📌 개발/운영 방침

- AI 모델 가중치 파일(`*.pth`, `*.h5`, `*.pt`)은 Git에 업로드되지 않는다.
- `.gitignore`에 다음 규칙을 추가하여 모델 가중치 파일을 자동으로 제외하였다.
  
```bash
/app/models/*/*.pth
/app/models/*/*.h5
/app/models/*/*.pt
```

- 모델별(`model_a`, `model_b`)로 가중치와 서비스 코드(로더, 예측기)를 세트로 관리한다.
- WebSocket 기능을 통해 프론트엔드에서 전달하는 **이미지 프레임**을 수신하고, AI 모델로 실시간 분석을 수행할 예정이다.
- 프로젝트 확장성(모델 추가/교체)을 고려하여 Dispatcher 구조(`predict_service.py`)를 설계하였다.

---

# 📌 향후 확장 계획

| 기능 | 설명 |
|:---|:---|
| 다양한 AI 모델 추가 | 새로운 AI 모델을 추가하고, `/services/`, `/models/` 폴더를 통해 쉽게 확장 가능 |
| WebSocket 처리 고도화 | WebSocket으로 수신된 이미지 프레임을 다양한 모델로 분기 처리하고, 예측 결과를 빠르게 반환하는 최적화 작업 |
| FastAPI 구조 고도화 | 필요시 예외처리(Exception Handling), 미들웨어(Middleware) 추가 등을 통해 서버 품질 개선 예정 |

---

# 📌 WebSocket 작업 시 참고사항

- WebSocket은 HTTP 방식과 별도로 동작한다 (`@router.websocket()` 사용).
- 클라이언트는 서버로 **이미지 프레임**을 실시간으로 전송하고, 서버는 이를 AI 모델로 분석 후 응답한다.
- WebSocket 통신 중 에러나 연결 종료(Disconnect)를 처리할 수 있도록 반드시 try/except 구문을 사용한다.
- `services/ws/connection_manager.py` 파일을 통해 클라이언트 연결 및 메시지 송수신을 관리한다.
- 각 클라이언트 연결은 고유한 `client_id`로 식별된다.
- WebSocket 관련 라우터는 `api/v1/ws_routes.py` 파일에 작성한다.
- WebSocket 전송 데이터는 크기와 빈도를 조절하여 서버 과부하를 방지한다.

---

# ✅ 요약

> 본 FastAPI 프로젝트는 다양한 AI 모델을 실험하고, 프론트엔드로부터 실시간 이미지를 수신하여 분석하는 기능을 지원하기 위해 설계되었습니다.  
> 프로젝트 구조는 유지보수성과 확장성을 고려하여 체계적으로 설정되었습니다.

---
