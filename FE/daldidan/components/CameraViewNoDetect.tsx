// daldidan/components/CameraViewNoDetect.tsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, AppState, Button, ActivityIndicator, Alert } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import * as SplashScreen from 'expo-splash-screen';

import { useObjectDetection } from '../hooks/useObjectDetection';
import DetectionOverlay from './DetectionOverlay'; // 실시간 탐지 결과 오버레이
import AppleButton from './AppleButton'; // 캡쳐 트리거 버튼 컴포넌트
import AppleHint from './AppleHint'; // 탐지되지 않았을 때 힌트 컴포넌트

// ★★★ useAnalysisApiHandler 훅 임포트 ★★★
// useAnalysisApiHandler.ts 파일에 이 훅 구현 코드가 있어야 합니다. (resetAnalysis, originalImageSize 반환 포함)
import { useAnalysisApiHandler } from '../hooks/useAnalysisApiHandler';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';
import AnalyzedResultOverlay from './AnalyzedResultOverlay';

// Splash 자동 종료 방지 (카메라 준비 완료 시 수동으로 hideAsync 호출)
SplashScreen.preventAutoHideAsync();import AppleProcessing from './AppleProcessing';

export default function CameraView() {
  // Vision Camera에서 후면 카메라 장치 가져오기
  const device = useCameraDevice('back');

  // 상태 정의: 화면 크기, 앱 상태, 권한 여부, splash 준비 완료 여부
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const [appState, setAppState] = useState('active');
  const [countdown, setCountdown] = useState<number | null>(null);
  const countdownTimer = useRef<NodeJS.Timeout | null>(null);
  const justReset = useRef(false);
  const [autoCaptureEnabled, setAutoCaptureEnabled] = useState(true);
  const lastCenterRef = useRef<{ x: number; y: number } | null>(null);
  const [hasPermission, setHasPermission] = useState(false);
  const [ready, setReady] = useState(false);

  // (현재 미사용) 카메라 뷰 캡처 용도 참조 객체
  const viewShotRef = useRef(null);

  // 분석 관련 API 훅 호출 (사진 분석 및 결과 관련 상태와 함수들)
  const {
    triggerAnalysis,
    isAnalyzing,
    analyzedResults,
    analysisError,
    originalImageSize,
    resetAnalysis,
  } = useAnalysisApiHandler();

  // 앱 상태 변화 감지 (ex. background → active 복귀 시 카메라 다시 켜기 등)
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      console.log('[AppState] changed:', nextAppState);
      setAppState(nextAppState);
    });
    return () => subscription.remove();
  }, []);
  
  // 권한 요청 로직 (초기 1회 실행)
  useEffect(() => {
    (async () => {
      const currentStatus = await Camera.getCameraPermissionStatus();
      console.log('[디버깅] 현재 권한 상태:', currentStatus); // not-determined, denied, authorized

      if (currentStatus !== 'granted') {
        const newStatus = await Camera.requestCameraPermission();
        console.log('[디버깅] 권한 요청 결과:', newStatus);
        setHasPermission(newStatus === 'granted');
      } else {
        setHasPermission(true);
      }
    })();
  }, []);

  // 카메라 포맷 설정: 가능한 최대 FPS 포맷 선택
  const format = device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats?.[0];
  const fps = format ? Math.min(60, format.maxFps) : 30;

  // 디버깅용 상태 출력
  useEffect(() => {
    console.log('[디버깅] device:', device);
    console.log('[디버깅] format:', format);
    console.log('[디버깅] hasPermission:', hasPermission);
    console.log('[디버깅] ready:', ready);
  }, [device, format, hasPermission, ready]);

  // 카메라 및 탐지 모델 관련 훅 호출
  const {
    detections,
    frameProcessor,
    cameraRef,
  } = useObjectDetection(format);
   const hasApple = detections.some((d) => d.class_id === 52);
  useEffect(() => {
    if (
      !hasApple ||
      isAnalyzing ||
      analyzedResults !== null ||
      countdown !== null ||
      !autoCaptureEnabled ||
      justReset.current
    ) return;
    startCountdownAndCapture();
  }, [detections, countdown, isAnalyzing, analyzedResults, autoCaptureEnabled]);

  // 카메라 준비 완료 시 splash 해제 및 렌더링 시작
  useEffect(() => {
    if (device && hasPermission && format) {
      SplashScreen.hideAsync();
      setReady(true);
    }
  }, [device, hasPermission, format]);

  // 전체 화면 캡쳐 및 API 요청 함수
  const handleCaptureAndAnalyze = useCallback(async () => {
    if (!cameraRef.current) {
      console.error("[Capture] Camera ref is not set.");
      Alert.alert("Error", "Camera not ready.");
      return;
    }
    if (isAnalyzing) {
      console.log("[API] Analysis already in progress. Skipping capture.");
      return;
    }

    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: 'speed',
        enableShutterAnimation: false,
      });

      const uri = `file://${photo.path}`;
      const photoOriginalWidth = photo.width;
      const photoOriginalHeight = photo.height;

      if (!uri || uri === 'file://undefined') {
        console.error("[Capture] Failed to capture photo, URI is null or invalid:", uri);
        Alert.alert("Error", "Failed to capture photo.");
        return;
      }

      await triggerAnalysis(uri, photoOriginalWidth, photoOriginalHeight);

    } catch (error: any) {
      console.error("[CameraView] Error during photo capture or triggering analysis:", error);
      Alert.alert("Analysis Failed", error.message || "An error occurred during analysis.");
    }
    setCountdown(null);
  }, [isAnalyzing, triggerAnalysis, cameraRef]);

  const startCountdownAndCapture = () => {
  if (countdownTimer.current || isAnalyzing || analyzedResults !== null) return;

  setCountdown(3); // 시작 숫자
  let current = 3;

  countdownTimer.current = setInterval(() => {
    current -= 1;
    if (current > 0) {
      setCountdown(current);
    } else {
      clearInterval(countdownTimer.current!);
      countdownTimer.current = null;
      
      handleCaptureAndAnalyze(); // 자동 캡처 실행
    }
  }, 1000);
};

  // AppleButton 또는 다른 캡쳐 트리거 UI 표시 여부 결정
  const appleOrDonutDetected = detections.some(d => d.class_id === 52 || d.class_id === 59);


 
  // 분석 완료 상태 판단: analyzedResults가 null이 아니고 배열이며, isAnalyzing이 false일 때
  const analysisFinished = analyzedResults !== null && !isAnalyzing;

  // 준비가 안 됐으면 splash 유지 (null 반환)
  if (!device || !hasPermission || !format || !ready) {
    return null;
  }

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
      {detections.length > 0 && !isAnalyzing && analyzedResults === null ? (
         <DetectionOverlay
           detections={detections}
           screenSize={screenSize} // 화면 크기 (onLayout 후 업데이트된 값)
           format={format}
          //  detectionResults={[]}
        />
      ) : null}


      {/* ★★★ API 분석 결과 오버레이 (AnalyzedResultOverlay) ★★★ */}
      {/* 분석 완료 상태이고 결과가 있으며, 원본 크기 정보가 있고, ★★★ 화면 크기도 유효할 때만 렌더링 ★★★ */}
      {/* screenSize가 0이 아니게 업데이트된 후에 이 조건이 true가 될 가능성이 생깁니다. */}
      {analysisFinished && analyzedResults && analyzedResults.length > 0 && originalImageSize && screenSize.width > 0 && screenSize.height > 0 ? (
          // ★★★ AnalyzedResultOverlay 컴포넌트 렌더링 ★★★
          <AnalyzedResultOverlay results={analyzedResults} screenSize={screenSize} originalImageSize={originalImageSize} />
      ) : null}

      {/* 결과 없음 메시지 */}
      {analysisFinished && analyzedResults?.length === 0 && (
        <View style={styles.noDetectionMessage}><Text style={styles.noDetectionText}>객체 인식 결과 없음</Text></View>
      )}


       {/* 캡쳐 버튼 등 나머지 UI 요소들 */}

       {/* 사과 또는 도넛 탐지 시 캡쳐 버튼 표시 */}
       {/* {appleOrDonutDetected && !isAnalyzing && analyzedResults === null ? (
          <View style={styles.captureButtonContainer}>
              <AppleButton
                  detections={detections}
                  onPress={handleCaptureAndAnalyze}
              />
          </View>
       ) : null}
     */}
      {countdown !== null && (
        <View style={styles.countdownOverlay}>
          <Text style={styles.countdownText}>
            {'🍎'.repeat(countdown)}
          </Text>
        </View>
      )}



       {/* 분석 중 인디케이터 표시 */}
      {isAnalyzing && (
      <AppleProcessing status="juicing" />
    )}

      {/* 아무 것도 탐지되지 않았을 때 힌트 표시 */}
      {detections.length === 0 && !isAnalyzing && analyzedResults === null && (
        <AppleHint />
      )}


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
                  <Button title="🐝사과 찾으러가기" onPress={() => {
                      resetAnalysis(); // 훅에서 가져온 resetAnalysis 함수 호출
                      setCountdown(null);
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
    countdownOverlay: {
      position: 'absolute',
      bottom: 120,
      alignSelf: 'center',   
      // backgroundColor: 'rgba(0, 0, 0, 0.6)',
      paddingVertical: 16,
      paddingHorizontal: 24,
      borderRadius: 20,
      zIndex: 50,
    },
countdownText: {
  fontSize: 48,
  fontWeight: 'bold',
  color: 'white',
},
});
