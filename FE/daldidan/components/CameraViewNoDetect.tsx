// daldidan/components/CameraViewNoDetect.tsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, AppState, Button, ActivityIndicator, Alert } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import * as SplashScreen from 'expo-splash-screen';

import { useObjectDetection } from '../hooks/useObjectDetection';
import AppleButton from './AppleButton';
import AppleHint from './AppleHint';
import { useAnalysisApiHandler } from '../hooks/useAnalysisApiHandler';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';
import AnalyzedResultOverlay from './AnalyzedResultOverlay';

// Splash 자동 종료 방지 (카메라 준비 완료 시 수동으로 hideAsync 호출)
SplashScreen.preventAutoHideAsync();

export default function CameraView() {
  // Vision Camera에서 후면 카메라 장치 가져오기
  const device = useCameraDevice('back');

  // 상태 정의: 화면 크기, 앱 상태, 권한 여부, splash 준비 완료 여부
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const [appState, setAppState] = useState('active');
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

  // 카메라 준비 완료 시 splash 해제 및 렌더링 시작
  useEffect(() => {
    if (device && hasPermission && format) {
      SplashScreen.hideAsync();
      setReady(true);
    }
  }, [device, hasPermission, format]);

  // 사과 탐지 시 사진 캡처 및 API 분석 요청 실행
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
  }, [isAnalyzing, triggerAnalysis, cameraRef]);

  // class_id = 52 (사과) 또는 59 (도넛) 이 탐지되었는지 여부
  const appleOrDonutDetected = detections.some(d => d.class_id === 52 || d.class_id === 59);

  // 분석이 완료되었는지 여부 판단
  const analysisFinished = analyzedResults !== null && !isAnalyzing;

  // 준비가 안 됐으면 splash 유지 (null 반환)
  if (!device || !hasPermission || !format || !ready) {
    return null;
  }

  return (
    <View style={StyleSheet.absoluteFill} onLayout={(event) => {
      const { width, height } = event.nativeEvent.layout;
      setScreenSize({ width, height });
      console.log('[CameraView] screenSize updated:', { width, height });
    }}>

      {/* 카메라 출력 */}
      {appState === 'active' && (
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
      )}

      {/* 분석 결과 존재 시 오버레이 렌더 */}
      {analysisFinished && analyzedResults?.length > 0 && originalImageSize && screenSize.width > 0 && screenSize.height > 0 && (
        <AnalyzedResultOverlay results={analyzedResults} screenSize={screenSize} originalImageSize={originalImageSize} />
      )}

      {/* 결과 없음 메시지 */}
      {analysisFinished && analyzedResults?.length === 0 && (
        <View style={styles.noDetectionMessage}><Text style={styles.noDetectionText}>객체 인식 결과 없음</Text></View>
      )}

      {/* 사과/도넛 탐지되었을 때 캡처 버튼 표시 */}
      {appleOrDonutDetected && !isAnalyzing && analyzedResults === null && (
        <View style={styles.captureButtonContainer}>
          <AppleButton detections={detections} onPress={handleCaptureAndAnalyze} />
        </View>
      )}

      {/* 분석 중일 때 로딩 인디케이터 표시 */}
      {isAnalyzing && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#ffffff" />
          <Text style={styles.loadingText}>분석 중...</Text>
        </View>
      )}

      {/* 아무 것도 탐지되지 않았을 때 힌트 표시 */}
      {detections.length === 0 && !isAnalyzing && analyzedResults === null && (
        <AppleHint />
      )}

      {/* 분석 완료 후 다시 시작 버튼 표시 */}
      {analysisFinished && (
        <View style={styles.resumeButtonContainer}>
          <Button title="다시 시작" onPress={resetAnalysis} />
        </View>
      )}
    </View>
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
});
