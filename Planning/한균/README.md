# 2025-04-23

## 아이디어 구체화 및 중간 발표 준비

### 🍏 달디단 – 사과 실시간 당도 예측 앱 (서버 추론 기반 MVP)

#### 📌 발표 시작
- A, B, C 등급의 빨간 사과 사진 3장을 제시하며 시작
- 겉보기엔 비슷하지만 당도가 다르다는 점을 강조

#### 🎯 기획 의도
- 기존 당도 측정은 고가 장비 필요 → 일반 사용자 접근 어려움
- 스마트폰 카메라로 누구나 간편하게 당도 예측 → 맛있는 과일 구매 도우미

#### 📖 기술적 근거
- RGB(색상값)와 당도(Brix) 간 상관관계 확인 (논문 및 보고서 인용)
- 색상 분석 기반 당도 예측 가능성 제시

#### 🧠 구현 계획 및 흐름
1. **데이터셋 확보**: 전북 장수 사과 당도 품질 데이터 (50만 장)
   - AI Hub 공개 데이터 활용
   - 기존 모델은 사과 등급 분류용 → 회귀 문제로 리모델링 필요

2. **실시간 추론 흐름**
   - getUserMedia로 카메라 뷰 띄우기
   - 1초마다 canvas로 프레임 캡처 + 압축 (toDataURL 또는 toBlob)
   - WebSocket을 통한 이미지 전송 및 결과 수신

3. **프론트엔드 예시 코드**
```jsx
const video = document.querySelector("video");
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
  video.play();
});

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
canvas.width = 224;
canvas.height = 224;

setInterval(() => {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL("image/jpeg", 0.5);
  sendImageToBackend(dataURL);
}, 1000);

const socket = new WebSocket("ws://localhost:8000/ws");
function sendImageToBackend(dataURL) {
  const base64 = dataURL.split(',')[1];
  socket.send(base64);
}
socket.onmessage = (e) => {
  const result = JSON.parse(e.data);
  console.log("예측 결과:", result);
};
```

#### 🔬 사용 기술 및 도구
| 항목 | 내용 |
|------|------|
| 데이터셋 | 전북 장수 사과 당도 품질 데이터 (AI Hub) |
| 실시간 영상 | getUserMedia + Canvas 압축 |
| 통신 방식 | WebSocket |
| 서버 추론 | FastAPI + PyTorch 모델 (Mask-RCNN 기반 재활용) |
| 흐림 감지 | OpenCV.js Laplacian Variance 방식 적용 |

#### 🎨 UI/UX 고도화 계획
- 흔들어서 측정 시작 기능
- 당도에 따라 햅틱 진동 제공
- AR 기반 시각화 인터페이스 적용 예정

#### ✅ MVP 활용 시나리오
- 마트에서 사과 구매 전 → 스마트폰 카메라로 당도 확인
- UX 데모 영상 or 생성형 AI 기반 시연 영상 활용 가능

#### 🌱 향후 확장 계획
- RGB 외 요소 분석 (채도, 명도 등)
- 다양한 과일 추가 대응
- 당도 예측 정확도 개선 및 지속적 학습

---
