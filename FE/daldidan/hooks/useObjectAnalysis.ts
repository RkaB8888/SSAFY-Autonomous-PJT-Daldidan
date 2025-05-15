// daldidan/hooks/useObjectAnalysis.ts

// API 응답 타입 임포트 또는 정의 (동일하게 유지)
import { AnalyzedObjectResult } from './types/objectDetection';
import { API_ENDPOINTS } from '../constants/api'; // API 엔드포인트 임포트

export const useObjectAnalysis = () => {
  // 기존 주석 코드는 이미 이전 수정에서 제거했다고 가정합니다.

  /**
   * 준비된 FormData (이미지 파일 URI 포함)를 받아 API에 POST 요청을 보내고,
   * 백엔드로부터 탐지된 객체 정보 (id, bbox, 당도 등)를 받아옵니다.
   * @param formData API 요청에 사용할 FormData 객체 (이미지 파일 및 기타 데이터 포함)
   * @returns API 응답으로 받은 분석 결과 배열 Promise
   * @throws Error API 요청 실패 시 예외 발생
   */
  const sendAnalysisRequest = async ( // 함수 이름 변경 (analyzeScreenshot -> sendAnalysisRequest)
    formData: FormData // FormData 객체를 인자로 받음
  ): Promise<AnalyzedObjectResult[]> => { // AnalyzedObjectResult[] 타입 반환

    console.log('[API] Sending analysis request with FormData...');

    try {
      // 이 엔드포인트가 multipart/form-data를 받는 백엔드 API 주소여야 합니다.
      const response = await fetch(API_ENDPOINTS.OBJECT_ANALYSIS, {
        method: 'POST',
        body: formData, // FormData 객체 전달
        // FormData 사용 시 'Content-Type' 헤더는 fetch가 자동으로 설정합니다.
        // 따라서 별도로 'Content-Type': 'multipart/form-data' 헤더를 설정할 필요 없습니다.
      });

      if (!response.ok) {
        const errorBody = await response.text();
        console.error(`[API] HTTP Error: ${response.status} ${response.statusText}`, errorBody);
        throw new Error(`API request failed with status ${response.status}: ${errorBody}`);
      }

      // 백엔드 응답 JSON 파싱
      const result: AnalyzedObjectResult[] = await response.json();

      console.log('[API] Analysis Result Received:', result);

      // Optional: 응답 구조 검증 (필요시 유지)

      return result; // 분석 결과 배열 반환

    } catch (error: any) {
      console.error('[API] Analysis API request failed:', error);
      throw error; // 호출하는 컴포넌트에서 처리하도록 오류 다시 던지기
    }
  };

  return {
    // 함수 이름 변경하여 노출
    // analyzeScreenshot, // 기존 함수 (이제 필요 없음)
    sendAnalysisRequest, // 새 함수
  };
};