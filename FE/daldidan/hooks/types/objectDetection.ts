// daldidan/hooks/types/objectDetection.ts

// Vision Camera 프레임 프로세서에서 사용되는 탐지 결과 타입
export interface Detection {
  x: number; // Vision Camera frame processor 좌표계 기준 (모델 입력과 다를 수 있음)
  y: number;
  width: number;
  height: number;
  score?: number;
  class_id?: number;
  // sugar_content 필드는 이제 API 응답에서 오므로 여기서는 제거하거나 옵션으로 둡니다.
  // sugar_content?: number;
}

// 백엔드 API에서 받아올 객체 탐지 및 분석 결과의 타입 정의
// 이 타입은 useObjectAnalysis, useAnalysisApiHandler, CameraViewNoDetect 등에서 사용됩니다.
export interface AnalyzedObjectResult {
  id?: number; // 백엔드에서 부여한 고유 ID 또는 클래스 ID 등 (API 명세 확인)
  class_id: number; // 객체 클래스 ID (예: 52 for apple, 59 for donut)
  label: string; // 객체 라벨 (예: "apple", "donut")
  bbox: { // 백엔드에서 분석한 바운딩 박스 좌표 (API 명세 확인, 보통 원본 이미지 기준)
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  // score?: number; // 백엔드에서 온 탐지 신뢰도 점수 (있다면)
  sugar_content?: number | null; // 사과의 경우 당도 값 (없거나 측정 실패 시 null 또는 undefined)
  // 도넛의 경우 다른 분석 값 필드가 필요할 수 있습니다. 예: sweetness?: number | null;
  // ... 백엔드 응답 JSON 구조에 정확히 맞춰 필드 추가 ...
}

// API 응답이 분석된 객체 결과들의 배열이라고 가정
export type ScreenshotAnalysisResponse = AnalyzedObjectResult[];