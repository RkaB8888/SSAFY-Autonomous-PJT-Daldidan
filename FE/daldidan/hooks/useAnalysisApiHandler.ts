// daldidan/hooks/useAnalysisApiHandler.ts

import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import { useObjectAnalysis } from './useObjectAnalysis'; // sendAnalysisRequest 함수를 가진 훅 임포트
import { AnalyzedObjectResult } from './types/objectDetection'; // API 응답 타입 임포트

export const useAnalysisApiHandler = () => {
    const { sendAnalysisRequest } = useObjectAnalysis();

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analyzedResults, setAnalyzedResults] = useState<AnalyzedObjectResult[] | null>(null);
    const [analysisError, setAnalysisError] = useState<string | null>(null);

    /**
     * 분석 결과 및 에러 상태를 초기 상태로 되돌립니다.
     * 카메라를 다시 켜고 실시간 탐지 모드로 돌아가는 데 사용됩니다.
     */
    const resetAnalysis = useCallback(() => {
        console.log("[useApiHandler] Resetting analysis states.");
        setAnalyzedResults(null);
        setAnalysisError(null);
        // isAnalyzing은 분석이 끝났으면 이미 false일 것임.
    }, []); // 의존성 배열 비워둡니다.

    /**
     * 캡쳐된 이미지 URI를 받아 API 분석을 트리거하고 상태를 업데이트하는 함수.
     */
    const triggerAnalysis = useCallback(async (uri: string): Promise<AnalyzedObjectResult[] | undefined> => {
        if (isAnalyzing) {
            console.log("[useApiHandler] Analysis already in progress.");
            return undefined;
        }

        setIsAnalyzing(true);
        setAnalyzedResults(null);
        setAnalysisError(null);
        console.log("[useApiHandler] Starting analysis for URI:", uri);

        try {
            const formData = new FormData();
            formData.append('image', {
              uri: uri,
              name: `screenshot_${Date.now()}.jpg`,
              type: 'image/jpeg',
            } as any);

            // TODO: 필요하다면 이미지 외 다른 데이터 추가

            const results = await sendAnalysisRequest(formData);

            console.log("[useApiHandler] Analysis Results Received:", results);
            setAnalyzedResults(results);
            setAnalysisError(null);

            return results;

        } catch (error: any) {
            console.error("[useApiHandler] Analysis Error:", error);
            setAnalysisError(error.message || "An unknown error occurred.");
            setAnalyzedResults(null); // 에러 발생 시 결과 초기화
            Alert.alert("Analysis Failed", error.message || "An unknown error occurred.");

            throw error;

        } finally {
            setIsAnalyzing(false);
            console.log("[useApiHandler] Analysis process finished.");
        }
    }, [isAnalyzing, sendAnalysisRequest]); // 의존성 배열

    return {
        triggerAnalysis,
        isAnalyzing,
        analyzedResults,
        analysisError,
        resetAnalysis, // ★★★ 초기화 함수를 반환 목록에 추가 ★★★
    };
};