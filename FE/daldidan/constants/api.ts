// API 엔드포인트 정의
export const API_BASE_URL = 'https://k12e206.p.ssafy.io'; // 개발 환경
// export const API_BASE_URL = 'https://api.yourdomain.com'; // 프로덕션 환경

export const API_ENDPOINTS = {
  OBJECT_ANALYSIS: `${API_BASE_URL}/predict`,
} as const;
