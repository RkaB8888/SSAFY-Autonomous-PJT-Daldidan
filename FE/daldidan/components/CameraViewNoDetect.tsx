// daldidan/components/CameraViewNoDetect.tsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, AppState, Dimensions, Button, ActivityIndicator, Alert } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera'; // Photo 타입 임포트
import { useObjectDetection } from '../hooks/useObjectDetection';
// import DetectionOverlay from './DetectionOverlay'; // 실시간 탐지 결과 오버레이
import AppleButton from './AppleButton'; // 캡쳐 트리거 버튼 컴포넌트
// ViewShot, captureRef는 이제 필요 없습니다.
// import ViewShot, { captureRef } from 'react-native-view-shot';
import AppleHint from './AppleHint'; // 탐지되지 않았을 때 힌트 컴포넌트

// ★★★ useAnalysisApiHandler 훅 임포트 ★★★
// useAnalysisApiHandler.ts 파일에 이 훅 구현 코드가 있어야 합니다. (resetAnalysis, originalImageSize 반환 포함)
import { useAnalysisApiHandler } from '../hooks/useAnalysisApiHandler';
// API 응답 타입 임포트 (훅 내부에서 관리되지만, 결과 오버레이에 전달)
import { AnalyzedObjectResult } from '../hooks/types/objectDetection'; // AnalyzedObjectResult 타입

// ★★★ API 분석 결과를 표시할 새로운 오버레이 컴포넌트 임포트 ★★★
// AnalyzedResultOverlay.tsx 파일에 구현 코드가 있어야 합니다. (이전 답변 코드 참고)
import AnalyzedResultOverlay from './AnalyzedResultOverlay'; // 임포트 주석 해제!
import AppleProcessing from './AppleProcessing';

export default function CameraView() {
  const device = useCameraDevice('back');
  // screenSize 상태는 onLayout 이벤트에서 업데이트됩니다. 초기값은 { width: 0, height: 0 }
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 }); // <-- 여기가 screenSize 선언 및 초기화
  const [appState, setAppState] = useState('active');

  // ViewShot ref는 takePhoto를 사용한다면 필요 없을 수 있습니다.
  const viewShotRef = useRef(null);


  // ★★★ useAnalysisApiHandler 훅 사용 ★★★
  // useAnalysisApiHandler.ts 파일에 이 훅 구현 코드가 있어야 합니다. (resetAnalysis, originalImageSize 반환 포함)
  const {
    triggerAnalysis, // API 분석 시작 함수 (훅 내부에서 FormData 생성 및 fetch 호출)
    isAnalyzing,     // API 분석 중 상태 (boolean)
    analyzedResults, // API 분석 완료된 결과 배열 (AnalyzedObjectResult[] | null)
    analysisError,   // API 에러 메시지 (string | null)
    originalImageSize, // ★★★ 훅에서 원본 이미지 해상도 상태 가져오기 (OriginalImageSize | null 타입) ★★★
    resetAnalysis,   // ★★★ 분석 결과 초기화 함수 (useAnalysisApiHandler 훅에서 반환 필요) ★★★
  } = useAnalysisApiHandler(); // 훅 호출

  // API 훅의 상태 (isAnalyzing, analyzedResults, analysisError)와 원본 해상도를 이 컴포넌트에서 직접 접근하여 UI 업데이트에 사용합니다.


  // App 상태 변화 감지
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      console.log('[AppState] changed:', nextAppState);
      setAppState(nextAppState);
    });
    return () => subscription.remove();
  }, []);

  // 카메라 설정 및 권한
  const format =
    device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats[0];
  const fps = format ? Math.min(60, format.maxFps) : 30;

  const {
    hasPermission,
    detections, // useObjectDetection에서 실시간으로 받아오는 탐지 결과
    frameProcessor, // useObjectDetection에서 정의된 프레임 프로세서 함수
    cameraRef, // useVisionCamera의 Camera 컴포넌트 ref (사진 촬영에 사용!)
    // detectionResults, // useObjectDetection에서 사용하던 예전 로직 (이제 사용 안 함)
  } = useObjectDetection(format);


  // 전체 화면 캡쳐 및 API 요청 함수
  const handleCaptureAndAnalyze = useCallback(async () => {
    // cameraRef를 takePhoto에 사용
    if (!cameraRef.current) {
      console.error("[Capture] Camera ref is not set.");
      Alert.alert("Error", "Camera not ready.");
      return;
    }
    // isAnalyzing 상태는 useAnalysisApiHandler 훅에서 관리되며, 훅 내부에서 중복 실행 방지됩니다.
    if (isAnalyzing) { // 훅에서 가져온 isAnalyzing 사용
        console.log("[API] Analysis already in progress. Skipping capture.");
        return;
    }

    // 분석 시작 시 useAnalysisApiHandler 내부에서 isAnalyzing 상태가 true로 변경됩니다.
    // 이 상태 변경을 Camera 컴포넌트의 isActive prop이 감지하여 카메라가 멈춥니다.

    try {
      console.log('[CameraView] Starting photo capture...');

      // cameraRef.current.takePhoto() 메서드를 사용하여 카메라 영상 캡처
      const photo = await cameraRef.current.takePhoto({
          qualityPrioritization: 'speed', // 속도 우선
          enableShutterAnimation: false, // 셔터 애니메이션 비활성화
      });

      const uri = `file://${photo.path}`;
      const photoOriginalWidth = photo.width; // 캡처된 원본 이미지 너비
      const photoOriginalHeight = photo.height; // 캡처된 원본 이미지 높이


      if (!uri || uri === 'file://undefined') {
         console.error("[Capture] Failed to capture photo, URI is null or invalid:", uri);
         Alert.alert("Error", "Failed to capture photo.");
         return;
      }

      console.log(`[Capture] Photo captured to file URI: ${uri} (Resolution: ${photoOriginalWidth}x${photoOriginalHeight})`);

      // useAnalysisApiHandler 훅의 triggerAnalysis 함수 호출
      // 캡처된 사진 파일의 URI와 원본 해상도 정보를 함께 훅으로 전달
      await triggerAnalysis(uri, photoOriginalWidth, photoOriginalHeight);

      console.log("[CameraView] Triggered analysis process.");

    } catch (error: any) {
      console.error("[CameraView] Error during photo capture or triggering analysis:", error);
      Alert.alert("Analysis Failed", error.message || "An error occurred during analysis.");

    } finally {
       // 카메라 일시 정지/재개 로직은 isAnalyzing 상태에 의해 자동으로 처리됩니다.
    }
  }, [isAnalyzing, triggerAnalysis, cameraRef]);


  // AppleButton 또는 다른 캡쳐 트리거 UI 표시 여부 결정
  const appleOrDonutDetected = detections.some(d => d.class_id === 52 || d.class_id === 59);


 
  // 분석 완료 상태 판단: analyzedResults가 null이 아니고 배열이며, isAnalyzing이 false일 때
  const analysisFinished = analyzedResults !== null && !isAnalyzing;
 


  // ★★★ React 컴포넌트는 하나의 루트 엘리먼트만 반환해야 합니다. ★★★
  return (
    // View에 onLayout이 달려있고, 이 View가 화면 전체를 덮습니다.
    <View style={StyleSheet.absoluteFill} // 이 View가 화면 전체를 덮도록
      onLayout={(event) => {
        // ★★★ View의 레이아웃 정보가 확정되면 screenSize 상태 업데이트 ★★★
        // 이 부분이 setScreenSize를 호출하여 screenSize를 0이 아닌 실제 값으로 업데이트합니다.
        const { width, height } = event.nativeEvent.layout;
        setScreenSize({ width, height });
        console.log('[CameraView] screenSize updated:', { width, height }); // screenSize 업데이트 로그
      }}
    > 
 {!hasPermission || !device || !format ? (
      <View style={styles.container}>
        <Text style={{ color: 'white' }}>카메라 설정 또는 권한 확인 중...</Text>
      </View>
    ) : (
       <>
      {/* Camera 컴포넌트 */}
      {/* appState가 'active' 상태일 때만 Camera 마운트 */}
      {/* isAnalyzing 중이거나 analysisFinished 상태일 때 isActive는 false */}
      {appState === 'active' ? (
        <Camera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={!isAnalyzing && analyzedResults === null}
          frameProcessor={frameProcessor}
          fps={fps}
          format={format}
          photo={true}
        />
      ) : null}


      {/* 실시간 탐지 결과 오버레이 */}
      {/* {detections.length > 0 && !isAnalyzing && analyzedResults === null ? (
         <DetectionOverlay
           detections={detections}
           screenSize={screenSize} // 화면 크기 (onLayout 후 업데이트된 값)
           format={format}
           detectionResults={[]}
        />
      ) : null} */}


      {/* ★★★ API 분석 결과 오버레이 (AnalyzedResultOverlay) ★★★ */}
      {/* 분석 완료 상태이고 결과가 있으며, 원본 크기 정보가 있고, ★★★ 화면 크기도 유효할 때만 렌더링 ★★★ */}
      {/* screenSize가 0이 아니게 업데이트된 후에 이 조건이 true가 될 가능성이 생깁니다. */}
      {analysisFinished && analyzedResults && analyzedResults.length > 0 && originalImageSize && screenSize.width > 0 && screenSize.height > 0 ? (
          // ★★★ AnalyzedResultOverlay 컴포넌트 렌더링 ★★★
          <AnalyzedResultOverlay results={analyzedResults} screenSize={screenSize} originalImageSize={originalImageSize} />
      ) : null}


       {/* API 분석 완료 후 결과는 없지만 카메라는 정지 상태인 경우 */}
       {analysisFinished && analyzedResults !== null && analyzedResults.length === 0 ? (
            <View style={styles.noDetectionMessage}><Text style={styles.noDetectionText}>객체 인식 결과 없음</Text></View>
       ) : null}


       {/* 캡쳐 버튼 등 나머지 UI 요소들 */}

       {/* 사과 또는 도넛 탐지 시 캡쳐 버튼 표시 */}
       {appleOrDonutDetected && !isAnalyzing && analyzedResults === null ? (
          <View style={styles.captureButtonContainer}>
              <AppleButton
                  detections={detections}
                  onPress={handleCaptureAndAnalyze}
              />
          </View>
       ) : null}


       {/* 분석 중 인디케이터 표시 */}
      {isAnalyzing && (
      <AppleProcessing status="juicing" />
    )}


       {/* 탐지된 객체가 없을 때 힌트 메시지 */}
       {/* detections.length === 0 이고, isAnalyzing 중이 아니고, 분석 완료 상태가 아닐 때 표시 */}
       {detections.length === 0 && !isAnalyzing && analyzedResults === null ? (
          <AppleHint />
       ) : null}


        {/* analysisError 상태 표시 (필요시) */}
         {/* analysisError && !isAnalyzing ? (
             <View style={styles.errorOverlay}>
                  <Text style={styles.errorText}>Error: {analysisError}</Text>
             </View>
         ) : null */}


        {/* ★★★ 분석 완료 후 카메라를 다시 켜기 위한 버튼 등 UI 추가 필요 ★★★ */}
        {/* 분석 완료 상태일 때만 "다시 시작" 버튼 표시 */}
        {analysisFinished ? (
             <View style={styles.resumeButtonContainer}>
                  <Button title="다시 시작" onPress={() => {
                      resetAnalysis(); // 훅에서 가져온 resetAnalysis 함수 호출
                  }} />
             </View>
         ) : null}
      </>
    )}
  </View> // ✅ 여기 View 닫고
);     
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  grayedCamera: { opacity: 0.7 },
  captureButtonContainer: {
     position: 'absolute',
     bottom: 100,
     alignSelf: 'center',
     zIndex: 10,
  },
  loadingOverlay: {
     ...StyleSheet.absoluteFillObject,
     backgroundColor: 'rgba(0, 0, 0, 0.6)',
     justifyContent: 'center',
     alignItems: 'center',
     zIndex: 20,
  },
  loadingText: {
     color: 'white',
     marginTop: 10,
     fontSize: 16,
  },
   noDetectionMessage: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: [{ translateX: -100 }, { translateY: -50 }],
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    padding: 15,
    borderRadius: 10,
    zIndex: 5,
   },
   noDetectionText: {
     color: 'white',
     fontSize: 18,
     textAlign: 'center',
   },
    resumeButtonContainer: {
        position: 'absolute',
        bottom: 50,
        alignSelf: 'center',
        zIndex: 15,
    },
});