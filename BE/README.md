# 📚 BE (Backend) 서버 구조 설명

---

## 📌 프로젝트 개요

본 프로젝트는 FastAPI 기반의 **게이트웨이 + AI 추로 서버**로 구성되어 있고,

* 프로트엔드와는 **WebSocket(socket.io)** 통신을 통해 실시간 데이터 송수신
* AI 모델 추로는 로컴 FastAPI 서버(`ai/`)에서 담당하며, 모델을 dispatcher 방식으로 분기 처리
* 두 서버는 각각 동입적으로 실행되며, `be/`는 8000번 포트, `ai/`는 9000번 포트를 사용

---

## 📁 디렉토리 구조

```text
BE/
├── be/                                ← WebSocket + REST API 중계 서버
│   ├── main.py                        # be 서버 진입점 (port 8000)
│   ├── api/                           # REST API 라우트팅
│   ├── core/                          # 설정
│   ├── schemas/                       # 요청/응답 데이터 정의
│   ├── services/remote_model.py      # AI 추로 서버 연동
│   ├── socketio_app/                 # socket.io 이벤트 핸들링
│   └── utils/
│
├── ai/                                ← AI 추로/학습 서버
│   ├── main.py                        # ai 서버 진입점 (port 9000)
│   ├── api/                           # 추로용 REST API
│   ├── core/                          # 설정
│   ├── schemas/                       # 추로 요청/응답용
│   ├── common_utils/                 # 모델 공통 유틸 (예: bbox crop)
│   ├── validator/                    # 데이터셋 유향성 검증 도구
│   ├── dataset/                      # 학습용 이미지 + JSON (Git 무시됨)
│   │   ├── images/
│   │   └── jsons/
│   ├── services/
│   │   ├── model_a/
│   │   ├── model_b/
│   │   └── model_jhg1/               # LightGBM 기반 당도 추로 모델
│   │       ├── predict/              # 추로 로지크
│   │       ├── training/             # 학습 스크립트
│   │       ├── validation/           # 성능 평가
│   │       ├── utils/                # 특징 추출 등
│   │       └── weights/              # 모델 가운치 저장 (.gitignore 대상)
```

---

## 📌 주요 파일 및 역할

### 📂 ai/services/model\_jhg1/

| 경로                            | 설명                            |
| ----------------------------- | ----------------------------- |
| `training/train_lightgbm.py`  | 원본 이미지+JSON 기반 학습 파이프라인       |
| `predict/predictor.py`        | 학습된 모델로 당도 추로 수행              |
| `utils/feature_extractors.py` | 이미지 → 벡터 변환 (RGB, HSV, LBP 등) |
| `weights/lightgbm_model.pkl`  | 학습된 모델 가운치 (joblib)           |
| `validation/evaluate.py`      | 추로 결과 성능 평가 (RMSE, MAE 등)     |

---

## 📌 실행 방법 요약

```bash
# be 서버 실행 (포트 8000)
cd be
uvicorn main:app --reload --port 8000

# ai 서버 실행 (포트 9000)
cd ai
uvicorn main:app --reload --port 9000
```

* be Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
* ai Swagger: [http://localhost:9000/docs](http://localhost:9000/docs)

---

## 📌 .gitignore 주요 설정

```gitignore
# 학습 데이터셋 무시
ai/dataset/*
!ai/dataset/**/.gitkeep

# 전역 모델 가운치 무시
*.pkl
*.pt
*.h5
*.joblib
*.ckpt

# 빈 디렉토리 유지
!**/weights/.gitkeep
```

---

## 📌 be 서버 당도 예측 API

**POST /dummy\_predict**

| 필드      | 타입                               | 설명                |
| ------- | -------------------------------- | ----------------- |
| `id`    | int (form field)                 | 당도 평가를 위한 ID      |
| `image` | UploadFile (multipart/form-data) | 평가할 사과 이미지 (JPEG) |

**예시 curl 요청:**

```bash
curl -X 'POST' \
  'http://localhost:8000/dummy_predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'id=23' \
  -F 'image=@apple.jpg;type=image/jpeg'
```

**응답 경우:**

```json
{
  "id": 23,
  "sugar_content": 12.84
}
```

---

## 📄 WebSocket 구조 요약

* 프로트 → WebSocket → `image_data` 전송
* 게이트웨이 서버 → AI 서버에 `/predict` 요청
* AI 서버 → 예측 결과 반환 → WebSocket으로 클라이언트에 `prediction_result` 전송

---

## ✅ 요약

> 본 시스템은 실시간 사과 인심 → crop 이미지 추로 → 당도 예측까지 이어진 파이프라인을 FastAPI 기반으로 구현하였으며,
> Dispatcher 중심 설계를 통해 다양한 모델 교체 시험과 유지보수 모두 유연하게 대응할 수 있습니다.
