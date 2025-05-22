// daldidan/hooks/useObjectAnalysis.ts
import { API_ENDPOINTS } from '../constants/api';
import { AnalyzedObjectResult } from './types/objectDetection'; // API 응답 타입

export const useObjectAnalysis = () => {

  /**
   * 준비된 FormData (이미지 파일 URI 포함)를 받아 API에 POST 요청을 보냅니다.
   * 이 훅은 순수하게 API 호출 기능만 제공하며, 상태 관리는 하지 않습니다.
   * @param formData API 요청에 사용할 FormData 객체
   * @returns API 응답 Promise (성공 시 AnalyzedObjectResult 배열, 실패 시 예외 발생)
   * @throws Error HTTP 에러 또는 네트워크 에러 발생 시
   */
  const sendAnalysisRequest = async (
    formData: FormData
  ): Promise<AnalyzedObjectResult[]> => { // AnalyzedObjectResult[] 타입 반환
    console.log('FormDataaaaaaa', formData)
    console.log('[useObjectAnalysis] Sending analysis request...');
    
    try {
      // API_ENDPOINTS.OBJECT_ANALYSIS 주소가 백엔드 API와 일치하는지 확인
      const response = await fetch(API_ENDPOINTS.OBJECT_ANALYSIS, {
        method: 'POST',
        body: formData,
        // FormData 사용 시 'Content-Type' 헤더는 fetch가 자동으로 설정
      });
      console.log('formData:',formData)
      // HTTP 상태 코드가 2xx 범위인지 확인
      if (!response.ok) {
        const errorBody = await response.text();
        console.error(`[useObjectAnalysis] HTTP Error: ${response.status} ${response.statusText}`, errorBody);
        throw new Error(`API request failed with status ${response.status}: ${errorBody}`);
      }

      // 응답 본문을 JSON으로 파싱
      const result: AnalyzedObjectResult[] = await response.json();

      console.log('[useObjectAnalysis] Analysis Result Received:', result);

      // TODO: Optional: 응답 구조 검증 로직 추가 필요

      return result;

    } catch (error: any) {
      console.error('[useObjectAnalysis] API request failed:', error);
      throw error; // 호출자에게 오류 다시 던지기
    }
  };

  // 이 훅은 sendAnalysisRequest 함수만 노출합니다.
  return {
    sendAnalysisRequest,
  };
};