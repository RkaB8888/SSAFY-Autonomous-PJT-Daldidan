# 2025-04-25

## 중간 발표 피드백 반영 및 데이터 탐색

---

### 🍉 SweetFruit 확장 계획 – 다양한 과일 당도 예측으로 확장

#### 🎯 피드백 반영 방향
- 기존 사과 위주의 모델에서 수박, 감귤 등으로 예측 가능 품종 확장 요청 반영
- 다양한 데이터셋과 협력 기업 탐색을 통해 당도 예측의 범용성과 확장성 확보

---

### 🔍 학술적 근거 및 참고 문헌

- 색상과 당도의 상관관계에 대한 연구:
  - [DBpia 논문](https://www.dbpia.co.kr/pdf/cpViewer)
  - [KISS 논문](https://kiss.kstudy.com/Detail/Ar?key=3921221)
  - 감귤 관련 연구: [DBpia 감귤 논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10572552&utm_source=chatgpt.com)

---

### 📦 공개된 모델 및 데이터셋

#### 📌 수박 관련 AI 모델
- GitHub 예시: [Watermelon Eval (YOLO 기반)](https://github.com/crf0409/watermelon_eval)

#### 📊 데이터셋
- IEEE Dataport: [수박 이미지 + 당도 라벨링 데이터](https://ieee-dataport.org/documents/watermelon-appearance-and-knock-correlate-data-sets-sugar-content?utm_source=chatgpt.com)
- Kaggle: [Watermelon Dataset](https://www.kaggle.com/datasets/michael1448/watermelon/data)

---

### 🏭 협력 가능 기관 및 데이터 확보 경로

#### 1. **탑스팜(주)** (경남 김해)
- 국내 최초 AI 선별 시스템 도입
- 8개 카메라 기반 외관 촬영 + 비파괴 당도 측정기로 브릭스 데이터 생성
- 주요 유통처: 롯데마트, 롯데슈퍼
- **협력 가능성**: 산학협력 형태로 대규모 품질 데이터 제공 가능

#### 2. **롯데마트/롯데슈퍼 – 부산권 산지유통센터**
- AI 품질 판별 시스템 구축
- 산지 로컬MD가 상주하며 데이터 수집 및 관리
- **협의 가능성**: 품질관리팀을 통한 산학 제안 협의 가능

#### 3. **초록마을 – 부산권 물류/직영점**
- 일부 수박 품질 데이터 보유 (당도+이미지)
- 비파괴 측정 후 유통, 부산-경남권 중심 유통망 활용 가능
- **데이터 제공 가능성**: 별도 협의 필요

---

### 📬 제안 방식 및 협의 전략
| 업체 | 제안 방향 |
|------|------------|
| 탑스팜 | 연구 목적 명시 + 데이터 활용 계획 첨부 제안서 발송 |
| 롯데마트 | 산지유통센터 or 본사 품질관리팀에 산학협력 형태로 접근 |
| 초록마을 | 부산권 지점 or 본사 상품기획팀에 데이터 활용 제안 |

---

### ✅ 요약
- **확장 품종**: 사과 → 수박, 감귤 등
- **데이터 출처**: 공개 데이터셋 + 협력 업체 데이터
- **모델 확장**: YOLOv8 + 회귀 기반 모델 범용화 시도
- **비전**: 다양한 과일을 대상으로 휴대폰 하나로 맛있는 과일을 고를 수 있는 "소비자용 AI 품질 측정 플랫폼"으로 확장

---

