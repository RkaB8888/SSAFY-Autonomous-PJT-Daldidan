// daldidan/hooks/useAnalysisApiHandler.ts

import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import { useObjectAnalysis } from './useObjectAnalysis'; // sendAnalysisRequest 함수를 가진 훅 임포트
// API 응답 타입 임포트 (AnalyzedObjectResult는 객체 목록 타입, ScreenshotAnalysisResponse는 전체 응답 타입)
import { AnalyzedObjectResult, ScreenshotAnalysisResponse } from './types/objectDetection';

// 원본 이미지 해상도 타입 정의 (CameraViewNoDetect.tsx에서 전달받아 AnalyzedResultOverlay에 전달)
interface OriginalImageSize {
    width: number;
    height: number;
}

export const useAnalysisApiHandler = () => {
    const { sendAnalysisRequest } = useObjectAnalysis();

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    // analyzedResults는 API 응답의 'results' 배열만 저장합니다.
    const [analyzedResults, setAnalyzedResults] = useState<AnalyzedObjectResult[] | null>(null);
    const [analysisError, setAnalysisError] = useState<string | null>(null);
    // 원본 이미지 해상도 상태 추가 및 관리
    const [originalImageSize, setOriginalImageSize] = useState<OriginalImageSize | null>(null);


    /**
     * 분석 결과 및 에러 상태, 원본 이미지 해상도 상태를 초기 상태로 되돌립니다.
     */
    const resetAnalysis = useCallback(() => {
        console.log("[useApiHandler] Resetting analysis states.");
        setAnalyzedResults(null);
        setAnalysisError(null);
        setOriginalImageSize(null);
    }, []);

    /**
     * 캡쳐된 이미지 URI와 원본 해상도를 받아 API 분석을 트리거하고 상태를 업데이트하는 함수.
     * @param uri 캡쳐된 이미지 파일의 로컬 URI (file://...)
     * @param originalWidth 캡처된 원본 이미지의 너비
     * @param originalHeight 캡처된 원본 이미지의 높이
     * @returns Promise<AnalyzedObjectResult[] | undefined> 분석 완료 시 결과 배열 반환, 분석 중이거나 에러 발생 시 undefined 반환
     */
    const triggerAnalysis = useCallback(async (uri: string, originalWidth: number, originalHeight: number): Promise<AnalyzedObjectResult[] | undefined> => {
        if (isAnalyzing) {
            console.log("[useApiHandler] Analysis already in progress.");
            return undefined;
        }

        setIsAnalyzing(true);
        setAnalyzedResults(null); // 새 분석 시작 시 이전 결과 초기화
        setAnalysisError(null);
        setOriginalImageSize(null); // 새 분석 시작 시 원본 해상도 상태 초기화
        console.log(`[useApiHandler] Starting analysis for URI: ${uri} (Original Size: ${originalWidth}x${originalHeight})`);

        try {
            // ★★★ 원본 이미지 해상도 상태 저장 ★★★
            // 이 상태는 API 호출 성공/실패와 무관하게 캡처 성공 시 바로 설정
            setOriginalImageSize({ width: originalWidth, height: originalHeight });

            // FormData 객체 생성 및 파일 URI 추가
            const formData = new FormData();
            formData.append('image', {
              uri: uri,
              name: `screenshot_${Date.now()}.jpg`,
              type: 'image/jpeg',
            } as any);

            // TODO: 필요하다면 이미지 외 다른 데이터 추가

            // useObjectAnalysis 훅의 sendAnalysisRequest 함수 호출
            // sendAnalysisRequest는 API 응답 전체 (ScreenshotAnalysisResponse)를 반환해야 합니다.
            // 또는 sendAnalysisRequest 함수가 이미 응답에서 .results 배열만 추출해서 Promise<AnalyzedObjectResult[]>를 반환하도록 구현되어 있어야 합니다.
            // sendAnalysisRequest가 Promise<ScreenshotAnalysisResponse>를 반환한다고 가정하고 수정합니다.
            // 만약 sendAnalysisRequest가 이미 Promise<AnalyzedObjectResult[]>를 반환한다면 아래 코드 수정은 불필요합니다.

            // sendAnalysisRequest 함수가 Promise<ScreenshotAnalysisResponse>를 반환한다고 가정:
            const fullResponse: ScreenshotAnalysisResponse = await sendAnalysisRequest(formData);

            // ★★★ 응답 객체에서 'results' 배열만 추출하여 상태에 저장 ★★★
            const resultsArray = fullResponse.results; // ScreenshotAnalysisResponse 타입에서 .results 접근

            console.log("[useApiHandler] Analysis Results Received (extracted):", resultsArray);

            if (!Array.isArray(resultsArray)) {
                 console.error("[useApiHandler] API response 'results' is not an array:", fullResponse);
                 setAnalyzedResults([]); // 빈 배열로 설정하거나 null 설정
                 setAnalysisError("API 응답 형식이 올바르지 않습니다.");
                 Alert.alert("분석 실패", "API 응답 형식이 올바르지 않습니다.");
                 // throw new Error("Invalid API response format"); // 오류를 다시 던질 수도 있습니다.
                 return undefined; // 또는 빈 배열 반환
            }


            setAnalyzedResults(resultsArray); // ★★★ 추출한 배열을 상태에 저장 ★★★
            setAnalysisError(null); // 성공했으므로 에러 상태 초기화


            return resultsArray; // 추출한 배열을 호출하는 곳으로 반환 (CameraView에서 직접 사용 안 함)

        } catch (error: any) {
            console.error("[useApiHandler] Analysis Error:", error);
            setAnalysisError(error.message || "An unknown error occurred."); // 에러 state에 저장
            setAnalyzedResults(null); // 에러 발생 시 결과 초기화
            setOriginalImageSize(null); // 에러 발생 시 원본 해상도 초기화
            Alert.alert("분석 실패", error.message || "An unknown error occurred."); // 사용자에게 알림

            throw error; // 에러를 다시 던져서 호출하는 곳에서도 catch할 수 있게 하거나, 여기서만 처리

        } finally {
            setIsAnalyzing(false); // 분석 종료 상태로 변경 (성공/실패 무관)
            console.log("[useApiHandler] Analysis process finished.");
        }
    }, [isAnalyzing, sendAnalysisRequest]); // 의존성 배열

    // 훅이 반환하는 값들 (상태 및 트리거 함수)
    return {
        triggerAnalysis, // API 분석 시작 함수
        isAnalyzing,     // 분석 중 상태 (boolean)
        analyzedResults, // 분석 완료된 결과 배열 (AnalyzedObjectResult[] 또는 null)
        analysisError,   // 발생한 에러 메시지 (string 또는 null)
        originalImageSize, // ★★★ 원본 이미지 해상도 상태 반환 ★★★
        resetAnalysis, // 분석 결과 초기화 함수
    };
};