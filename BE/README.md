# 📚 BE (Backend) 서버 구조 설명

---

## 📌 프로젝트 개요

본 프로젝트는 FastAPI 기반의 **게이트웨이 + AI 추론 서버**로 구성되어 있으며,  
- 프론트엔드와는 **WebSocket(socket.io)** 통신을 통해 실시간 데이터 송수신  
- AI 모델 추론은 로컬 FastAPI 서버(`ai/`)에서 담당하며, 모델을 dispatcher 방식으로 분기 처리

---

## 📁 디렉토리 구조

```text
BE/
├── be/                                ← WebSocket + REST API 중계 서버
│   ├── main.py
│   ├── api/                           # REST API 라우팅
│   ├── core/                          # 설정
│   ├── schemas/                       # 요청/응답 데이터 정의
│   ├── services/remote_model.py      # AI 추론 서버 연동
│   ├── socketio_app/                 # socket.io 이벤트 핸들링
│   └── utils/
│
├── ai/                                ← AI 추론/학습 서버
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── schemas/                       # 추론 요청/응답용
│   ├── common_utils/                 # 모델 공통 유틸 (예: bbox crop)
│   ├── validator/                    # 데이터셋 유효성 검증 도구
│   ├── dataset/                      # 학습용 이미지 + JSON (Git 무시됨)
│   │   ├── images/
│   │   └── jsons/
│   ├── services/
│   │   ├── model_a/
│   │   ├── model_b/
│   │   └── model_jhg1/               # LightGBM 기반 당도 추론 모델
│   │       ├── predict/              # 추론 로직
│   │       ├── training/             # 학습 스크립트
│   │       ├── validation/           # 성능 평가
│   │       ├── utils/                # 특징 추출 등
│   │       └── weights/              # 모델 가중치 저장 (.gitignore 대상)
```

---

## 📌 주요 파일 및 역할

### 📂 ai/services/model_jhg1/

| 경로 | 설명 |
|------|------|
| `training/train_lightgbm.py` | 원본 이미지+JSON 기반 학습 파이프라인 |
| `predict/predictor.py` | 학습된 모델로 당도 추론 수행 |
| `utils/feature_extractors.py` | 이미지 → 벡터 변환 (RGB, HSV, LBP 등) |
| `weights/lightgbm_model.pkl` | 학습된 모델 가중치 (joblib) |
| `validation/evaluate.py` | 추론 결과 성능 평가 (RMSE, MAE 등) |

---

## 📌 .gitignore 주요 설정

```gitignore
# 학습 데이터셋 무시
ai/dataset/*
!ai/dataset/**/.gitkeep

# 전역 모델 가중치 무시
*.pkl
*.pt
*.h5
*.joblib
*.ckpt

# 빈 디렉토리 유지
!**/weights/.gitkeep
```

---

## 📌 AI 서버 예측 요청 API

**POST /api/v1/predict**

| 필드 | 타입 | 설명 |
|------|------|------|
| `model_name` | string | 사용할 모델 이름 (`model_jhg1`) |
| `image_base64` | string | base64로 인코딩된 사과 crop 이미지 |

**예시 요청:**
```json
{
  "model_name": "model_jhg1",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**예시 응답:**
```json
{
  "sugar_content": 13.52
}
```

---

## 📌 WebSocket 구조 요약

- 프론트 → WebSocket → `image_data` 전송  
- 게이트웨이 서버 → AI 서버에 `/predict` 요청  
- AI 서버 → 예측 결과 반환 → WebSocket으로 클라이언트에 `prediction_result` 전송

---

## ✅ 요약

> 본 시스템은 실시간 사과 인식 → crop 이미지 추론 → 당도 예측까지 이어지는 파이프라인을 FastAPI 기반으로 구현하였으며,  
> Dispatcher 중심 설계를 통해 다양한 모델 교체 실험과 유지보수 모두 유연하게 대응할 수 있습니다.
