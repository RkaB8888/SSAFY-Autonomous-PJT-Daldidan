// daldidan/components/CameraViewNoDetect.tsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  AppState,
  Dimensions,
  Button,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera'; // Photo 타입 임포트
import { useObjectDetection } from '../hooks/useObjectDetection';
// import DetectionOverlay from './DetectionOverlay'; // 실시간 탐지 결과 오버레이
import AppleHint from './AppleHint'; // 탐지되지 않았을 때 힌트 컴포넌트
// ★★★ useAnalysisApiHandler 훅 임포트 ★★★
// useAnalysisApiHandler.ts 파일에 이 훅 구현 코드가 있어야 합니다. (resetAnalysis, originalImageSize 반환 포함)
import { useAnalysisApiHandler } from '../hooks/useAnalysisApiHandler';
// API 응답 타입 임포트 (훅 내부에서 관리되지만, 결과 오버레이에 전달)

// ★★★ API 분석 결과를 표시할 새로운 오버레이 컴포넌트 임포트 ★★★
// AnalyzedResultOverlay.tsx 파일에 구현 코드가 있어야 합니다. (이전 답변 코드 참고)
import AnalyzedResultOverlay from './AnalyzedResultOverlay'; // 임포트 주석 해제!
import AppleProcessing from './AppleProcessing';
import { useShake } from '../hooks/useShake';
import * as SplashScreen from 'expo-splash-screen';
import Sound from 'react-native-sound';
import countdownAudio from '../assets/sounds/countdown.mp3';
import CaptureOverlay from './CaptureOverlay';
SplashScreen.preventAutoHideAsync(); // Splash 화면을 수동으로 제어하겠다는 선언

export default function CameraView() {
  const device = useCameraDevice('back');
  const appleSoundRef = useRef<Sound | null>(null);
  // screenSize 상태는 onLayout 이벤트에서 업데이트됩니다. 초기값은 { width: 0, height: 0 }
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 }); // <-- 여기가 screenSize 선언 및 초기화
  const [appState, setAppState] = useState('active');
  const [countdown, setCountdown] = useState<number | null>(null);
  const countdownTimer = useRef<NodeJS.Timeout | null>(null);
  const justReset = useRef(false);
  const [autoCaptureEnabled, setAutoCaptureEnabled] = useState(true);
  const lastCenterRef = useRef<{ x: number; y: number } | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const countdownSoundRef = useRef<Sound | null>(null);
  const [showCaptureImage, setShowCaptureImage] = useState(false);
  const capturingRef = useRef(false);
  const [freezeDetection, setFreezeDetection] = useState(false);
  const [minSugar, setMinSugar] = useState(11);

  const captureFrames = [
    {
      character: require('../assets/images/apple_char1.png'),
      camera: require('../assets/images/apple_capture.png'),
      message: '사과를 찾았어요!',
    },
    {
      character: require('../assets/images/apple_char2.png'),
      camera: require('../assets/images/apple_capture2.png'),
      message: '포즈 잡는중... \n카메라를 가만히 들고 있어주세요!',
    },
    {
      character: require('../assets/images/apple_char3.png'),
      camera: require('../assets/images/apple_capture3.png'),
      message: '애플~~',
    },
  ];

  // ★★★ useAnalysisApiHandler 훅 사용 ★★★
  // useAnalysisApiHandler.ts 파일에 이 훅 구현 코드가 있어야 합니다. (resetAnalysis, originalImageSize 반환 포함)
  const {
    triggerAnalysis, // API 분석 시작 함수 (훅 내부에서 FormData 생성 및 fetch 호출)
    isAnalyzing, // API 분석 중 상태 (boolean)
    analyzedResults, // API 분석 완료된 결과 배열 (AnalyzedObjectResult[] | null)
    analysisError, // API 에러 메시지 (string | null)
    originalImageSize, // ★★★ 훅에서 원본 이미지 해상도 상태 가져오기 (OriginalImageSize | null 타입) ★★★
    resetAnalysis, // ★★★ 분석 결과 초기화 함수 (useAnalysisApiHandler 훅에서 반환 필요) ★★★
  } = useAnalysisApiHandler(); // 훅 호출

  useEffect(() => {
    Sound.setCategory('Playback');
    const snd = new Sound(countdownAudio, Sound.MAIN_BUNDLE, (err) => {
      if (err) console.warn('Countdown sound load error', err);
    });
    countdownSoundRef.current = snd;
    return () => {
      snd.release();
    };
  }, []);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (showCaptureImage) {
      timer = setInterval(() => {
        setFrameIndex((i) => (i + 1) % captureFrames.length);
      }, 2000);
    } else {
      setFrameIndex(0);
    }
    return () => {
      clearInterval(timer);
    };
  }, [showCaptureImage]);

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

  const { hasPermission, detections, frameProcessor, cameraRef } =
    useObjectDetection(format);

  // 사과 감지 상태를 실시간으로 업데이트
  const [hasApple, setHasApple] = useState(false);
  const hasAppleRef = useRef(false); // 추가: 최신 hasApple 상태를 참조하기 위한 ref

  useEffect(() => {
    hasAppleRef.current = hasApple; // 최신 hasApple 상태 저장
  }, [hasApple]);

  useEffect(() => {
    const appleDetected = detections.some((d) => d.class_id === 52);
    setHasApple(appleDetected);
  }, [detections]);

  useEffect(() => {
    if (device && hasPermission && format) {
      SplashScreen.hideAsync(); // 준비가 끝난 순간에 splash를 닫음
    }
  }, [device, hasPermission, format]);

  useEffect(() => {
    // 사과가 감지되지 않았을 때 카운트다운 초기화
    if (!hasApple && countdown !== null) {
      setCountdown(null);
      if (countdownTimer.current) {
        clearInterval(countdownTimer.current);
        countdownTimer.current = null;
      }
    }
    if (
      !hasApple ||
      isAnalyzing ||
      analyzedResults !== null ||
      // countdown !== null ||
      !autoCaptureEnabled ||
      justReset.current
    )
      return;
    startCaptureSequence();
  }, [detections, hasApple, isAnalyzing, analyzedResults, autoCaptureEnabled]);

  // 전체 화면 캡쳐 및 API 요청 함수
  const handleCaptureAndAnalyze = useCallback(async () => {
    // cameraRef를 takePhoto에 사용
    if (!cameraRef.current) {
      console.error('[Capture] Camera ref is not set.');
      Alert.alert('Error', 'Camera not ready.');
      return;
    }
    // isAnalyzing 상태는 useAnalysisApiHandler 훅에서 관리되며, 훅 내부에서 중복 실행 방지됩니다.
    if (isAnalyzing) {
      // 훅에서 가져온 isAnalyzing 사용
      console.log('[API] Analysis already in progress. Skipping capture.');
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
        console.error(
          '[Capture] Failed to capture photo, URI is null or invalid:',
          uri
        );
        Alert.alert('Error', 'Failed to capture photo.');
        return;
      }

      console.log(
        `[Capture] Photo captured to file URI: ${uri} (Resolution: ${photoOriginalWidth}x${photoOriginalHeight})`
      );

      // useAnalysisApiHandler 훅의 triggerAnalysis 함수 호출
      // 캡처된 사진 파일의 URI와 원본 해상도 정보를 함께 훅으로 전달
      await triggerAnalysis(uri, photoOriginalWidth, photoOriginalHeight);
      console.log('[CameraView] Triggered analysis process.');
      setFreezeDetection(true);
    } catch (error: any) {
      console.error(
        '[CameraView] Error during photo capture or triggering analysis:',
        error
      );
      Alert.alert(
        'Analysis Failed',
        error.message || 'An error occurred during analysis.'
      );
    } finally {
      // 카메라 일시 정지/재개 로직은 isAnalyzing 상태에 의해 자동으로 처리됩니다.
    }
    setCountdown(null);
  }, [isAnalyzing, triggerAnalysis, cameraRef]);

  useEffect(() => {
    if (analyzedResults !== null && !isAnalyzing) {
      setFreezeDetection(false); // 분석 끝났을 때만 다시 감지 허용
    }
  }, [analyzedResults, isAnalyzing]);

  const startCaptureSequence = () => {
    if (
      isAnalyzing ||
      analyzedResults !== null ||
      showCaptureImage ||
      capturingRef.current ||
      freezeDetection ||
      !hasApple
    )
      return;

    capturingRef.current = true;
    setShowCaptureImage(true);
    setFreezeDetection(true);

    // 사과 감지 상태 모니터링을 위한 인터벌 설정
    const appleDetectionCheck = setInterval(() => {
      if (!hasAppleRef.current) {
        // 수정: hasApple 대신 hasAppleRef.current 사용
        // 사과가 감지되지 않으면 모든 동작 중단
        clearInterval(appleDetectionCheck);
        setShowCaptureImage(false);
        capturingRef.current = false;
        setFreezeDetection(false);
        if (countdownSoundRef.current) {
          countdownSoundRef.current.stop();
        }
      }
    }, 100);

    // 🟡 사운드 재생 후에 촬영
    countdownSoundRef.current?.play((success) => {
      clearInterval(appleDetectionCheck);
      if (success && hasAppleRef.current) {
        // 수정: hasApple 대신 hasAppleRef.current 사용
        handleCaptureAndAnalyze().then(() => {
          setShowCaptureImage(false);
          capturingRef.current = false;
          setFreezeDetection(false);
        });
      } else {
        setShowCaptureImage(false);
        capturingRef.current = false;
        setFreezeDetection(false);
        if (countdownSoundRef.current) {
          countdownSoundRef.current.stop();
        }
      }
    });
  };

  // AppleButton 또는 다른 캡쳐 트리거 UI 표시 여부 결정
  const appleOrDonutDetected = detections.some(
    (d) => d.class_id === 52 || d.class_id === 59
  );

  // 분석 완료 상태 판단: analyzedResults가 null이 아니고 배열이며, isAnalyzing이 false일 때
  const analysisFinished = analyzedResults !== null && !isAnalyzing;

  useShake(
    () => {
      if (analysisFinished) {
        console.log('[Shake] 감지됨 → 분석 초기화');
        justReset.current = true; // ✅ 자동 캡처 방지 플래그 ON
        resetAnalysis();

        // ✅ 일정 시간 후 자동 캡처 다시 허용
        setTimeout(() => {
          justReset.current = false;
          console.log('[Shake] 자동 캡처 재허용됨');
        }, 2000); // 2초 뒤에 자동 캡처 허용
      }
    },
    2.0,
    700
  );

  // ★★★ React 컴포넌트는 하나의 루트 엘리먼트만 반환해야 합니다. ★★★
  return (
    // View에 onLayout이 달려있고, 이 View가 화면 전체를 덮습니다.
    <View
      style={StyleSheet.absoluteFill} // 이 View가 화면 전체를 덮도록
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
          <ActivityIndicator size='large' color='white' />
          <Text style={{ color: 'white', marginTop: 12 }}>
            카메라 설정 또는 권한 확인 중...
          </Text>
        </View>
      ) : (
        <View style={StyleSheet.absoluteFill}>
          {/* Camera 컴포넌트 */}
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

          {analysisFinished &&
          analyzedResults &&
          analyzedResults.length > 0 &&
          originalImageSize &&
          screenSize.width > 0 &&
          screenSize.height > 0 ? (
            <AnalyzedResultOverlay
              results={analyzedResults}
              screenSize={screenSize}
              originalImageSize={originalImageSize}
              minSugar={minSugar}
              onChangeMinSugar={setMinSugar}
            />
          ) : null}

          {analysisFinished &&
          analyzedResults !== null &&
          analyzedResults.length === 0 ? (
            <View style={styles.noDetectionMessage}>
              <Text style={styles.noDetectionText}>객체 인식 결과 없음</Text>
            </View>
          ) : null}

          {isAnalyzing && <AppleProcessing status='juicing' />}

          {detections.length === 0 &&
          !isAnalyzing &&
          analyzedResults === null &&
          !showCaptureImage &&
          !freezeDetection ? (
            <AppleHint />
          ) : null}

          <CaptureOverlay
            visible={showCaptureImage}
            framePair={captureFrames[frameIndex]}
          />
        </View>
      )}
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    justifyContent: 'center', // 수직 중앙
    alignItems: 'center', // 수평 중앙
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
  captureOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    zIndex: 100,
  },
  captureImage: {
    width: 200,
    height: 200,
  },
});
