// 클라이언트 → 서버로 보내는 데이터 타입
export interface FrameData {
    image: string; // base64 인코딩된 이미지 데이터
  }
  
export type ApplePrediction = {
  id: number;
  brix: number;
  box: [number, number, number, number]; // x, y, w, h
};

export type PredictionData = {
  results: ApplePrediction[];
};