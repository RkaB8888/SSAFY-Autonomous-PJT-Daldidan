export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  score?: number;
  class_id?: number;
  sugar_content?: number;
}

// export interface DetectionResult {
//   detection: Detection;
//   imageData: string;
//   result?: any;
//   timestamp: number;
// }

// export interface CroppedImageData {
//   data: number[];
//   width: number;
//   height: number;
//   isJPEG?: boolean;
// }

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
  score?: number; // 백엔드에서 온 탐지 신뢰도 점수 (있다면)
  sugar_content?: number | null; // 사과의 경우 당도 값 (없거나 측정 실패 시 null 또는 undefined)
  // 도넛의 경우 다른 분석 값 필드가 필요할 수 있습니다. 예: sweetness?: number | null;
  // ... 백엔드 응답 JSON 구조에 정확히 맞춰 필드 추가 ...
}

// API 응답이 분석된 객체 결과들의 배열이라고 가정
export type ScreenshotAnalysisResponse = AnalyzedObjectResult[];