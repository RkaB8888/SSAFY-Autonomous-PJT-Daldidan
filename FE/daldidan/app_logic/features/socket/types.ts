// 클라이언트 → 서버로 보내는 데이터 타입
export interface FrameData {
    image: string; // base64 인코딩된 이미지 데이터
  }
  
  // 서버 → 클라이언트로 받는 데이터 타입
  export interface PredictionData {
    brix: number;  // 예측된 당도 값
  }