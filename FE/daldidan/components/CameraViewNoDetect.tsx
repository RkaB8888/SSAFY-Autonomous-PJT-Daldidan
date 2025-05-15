// daldidan/components/CameraViewNoDetect.tsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { StyleSheet, Text, View, AppState, Dimensions, Button, ActivityIndicator, Alert } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useObjectDetection } from '../hooks/useObjectDetection';
// import DetectionOverlay from './DetectionOverlay'; // 실시간 탐지 결과 오버레이
import AppleButton from './AppleButton'; // 캡쳐 트리거 버튼 컴포넌트
import ViewShot, { captureRef } from 'react-native-view-shot'; // 화면 캡쳐 라이브러리
import AppleHint from './AppleHint'; // 탐지되지 않았을 때 힌트 컴포넌트

// ★★★ useAnalysisApiHandler 훅 임포트 ★★★
import { useAnalysisApiHandler } from '../hooks/useAnalysisApiHandler';
// API 응답 타입은 훅 내부에서 관리되지만, 결과 오버레이에 전달
import { AnalyzedObjectResult } from '../hooks/types/objectDetection'; // API 응답 타입

// ★★★ API 분석 결과를 표시할 새로운 오버레이 컴포넌트 임포트 ★★★
// 이 컴포넌트를 위에서 새로 만들었습니다.
import AnalyzedResultOverlay from './AnalyzedResultOverlay';

import AppleProcessing from './AppleProcessing';
import AppleBar from './AppleBar';

export default function CameraView() {
  const device = useCameraDevice('back');
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const [appState, setAppState] = useState('active');

  const viewShotRef = useRef(null);
  // cameraPaused state는 이제 useAnalysisApiHandler 훅의 isAnalyzing 및 analyzedResults 상태로 대체됩니다.


  // ★★★ useAnalysisApiHandler 훅 사용 ★★★
  // useAnalysisApiHandler.ts 파일에 이 훅의 구현 코드가 있어야 합니다.
  const {
    triggerAnalysis, // API 분석 시작 함수 (훅 내부에서 FormData 생성 및 fetch 호출)
    isAnalyzing,     // API 분석 중 상태 (boolean)
    analyzedResults, // API 분석 완료된 결과 배열 (AnalyzedObjectResult[] | null)
    analysisError,   // API 에러 메시지 (string | null)
    resetAnalysis,   // ★★★ 분석 결과 초기화 함수 (useAnalysisApiHandler 훅에서 반환 필요) ★★★
  } = useAnalysisApiHandler(); // 훅 호출

  // API 훅의 상태 (isAnalyzing, analyzedResults, analysisError)는 이 컴포넌트에서 직접 접근하여 UI 업데이트에 사용합니다.


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
    cameraRef, // useVisionCamera의 Camera 컴포넌트 ref
    // detectionResults, // useObjectDetection에서 사용하던 예전 로직 (이제 사용 안 함)
  } = useObjectDetection(format);


  // 전체 화면 캡쳐 및 API 요청 함수
  const handleCaptureAndAnalyze = useCallback(async () => {
    if (!viewShotRef.current) {
      console.error("[Capture] ViewShot ref is not set.");
      Alert.alert("Error", "Could not capture screen.");
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
      console.log('[CameraView] Starting screen capture...');

      // ViewShot 캡쳐 (파일 URI 형식으로)
      // result: 'file' 또는 'tmpfile' 사용
      const uri = await captureRef(viewShotRef, {
        format: "jpg", // 또는 "jpg"
        quality: 0.9,
        result: "tmpfile", // 또는 "tmpfile"
      });

      if (!uri) {
         console.error("[Capture] Failed to capture screen, URI is null or empty.");
         Alert.alert("Error", "Failed to capture screen.");
         return; // 캡쳐 실패 시 중단
      }

      console.log("[Capture] Screen captured to file URI:", uri);

      // ★★★ useAnalysisApiHandler 훅의 triggerAnalysis 함수 호출 ★★★
      // 이 함수 내부에서 FormData 생성, API 요청, 상태 업데이트 (isAnalyzing, analyzedResults, analysisError) 모두 처리됩니다.
      await triggerAnalysis(uri); // 훅에서 가져온 함수 호출

      console.log("[CameraView] Triggered analysis process.");

      // 분석 완료 및 에러 처리는 useAnalysisApiHandler 훅 내부 및 훅이 반환하는 상태를 통해 이루어집니다.
      // 분석 완료 후 카메라 정지 상태 (isActive=false)는 isAnalyzing=false가 되면서 유지됩니다. (아래 isActive 로직 참고)

    } catch (error: any) {
      // triggerAnalysis에서 발생하고 다시 던져진 에러를 여기서 catch할 수도 있지만,
      // 훅 내부에서 이미 Alert 등으로 사용자에게 알림을 처리하고 있다면 불필요할 수 있습니다.
      console.error("[CameraView] Error during capture or triggering analysis:", error);
      // Alert.alert("Capture/Trigger Error", error.message || "An error occurred."); // 중복 알림 주의
    } finally {
       // 카메라 일시 정지/재개 로직은 isAnalyzing 상태에 의해 자동으로 처리됩니다.
       // 별도의 setCameraPaused(false) 호출이 필요 없습니다.
    }
  }, [isAnalyzing, triggerAnalysis]); // 의존성 배열: isAnalyzing, triggerAnalysis


  // AppleButton 또는 다른 캡쳐 트리거 UI 표시 여부 결정
  // useObjectDetection에서 오는 실시간 detections를 사용
  // 사과 (class_id 52) 또는 도넛 (class_id 59)이 탐지되면 버튼 표시
  const appleOrDonutDetected = detections.some(d => d.class_id === 52 || d.class_id === 59);


  // 카메라 권한 또는 설정 로딩 실패 시
  if (!hasPermission || !device || !format) {
    return <View style={styles.container}><Text style={{color:'white'}}>카메라 설정 또는 권한 확인 중...</Text></View>;
  }

  // 분석 완료 상태 판단: analyzedResults가 null이 아니고 배열이며, isAnalyzing이 false일 때
  const analysisFinished = analyzedResults !== null && !isAnalyzing;


  // ★★★ React 컴포넌트는 하나의 루트 엘리먼트만 반환해야 합니다. ★★★
  // ViewShot으로 캡쳐할 영역과 그 위에 표시될 오버레이, 버튼 등을 하나의 <React.Fragment>로 감쌉니다.
  return (
    <React.Fragment> {/* 최상위 엘리먼트로 React.Fragment 사용 */}
      {/* ViewShot으로 캡쳐할 전체 영역 */}
       <ViewShot ref={viewShotRef} style={StyleSheet.absoluteFill} options={{ format: 'jpg', quality: 0.9 }}>
         {/* 캡쳐 대상 영역 전체 View */}
         <View
           style={StyleSheet.absoluteFill} // 이 View가 ViewShot을 꽉 채우도록
           onLayout={(event) => {
             const { width, height } = event.nativeEvent.layout;
             setScreenSize({ width, height });
           }}
         >
           {/* Camera 컴포넌트 */}
           {/* appState가 'active' 상태일 때만 Camera 마운트 */}
           {/* isAnalyzing 중이거나 analysisFinished 상태일 때 isActive는 false */}
           {appState === 'active' ? ( // ★★★ 조건부 렌더링 수정: 삼항 연산자 사용 ★★★
             <Camera
               ref={cameraRef}
               style={StyleSheet.absoluteFill}
               device={device}
               // ★★★ 카메라 활성화 조건: App Active이고, 분석 중이 아니며, 분석 완료 상태가 아닐 때 ★★★
               // isAnalyzing 중이거나 analysisFinished 상태일 때 isActive는 false
               isActive={!isAnalyzing && analyzedResults === null} // appState === 'active' 조건은 이미 상위 View에서 체크
               frameProcessor={frameProcessor} // isActive가 false면 실행 안됨
               fps={fps}
               format={format}
               photo={true}
             />
           ) : null} {/* ★★★ 조건이 false일 때 null 반환 ★★★ */}


           {/*
              실시간 탐지 결과 오버레이 (CameraViewNoDetect에서 가져온 detections 사용)
              API 분석 중이 아니고 분석 결과가 없을 때만 실시간 오버레이를 보여줍니다.
           */}
           {/* {detections.length > 0 && !isAnalyzing && analyzedResults === null ? ( // isAnalyzing 중이 아니고 분석 결과가 없을 때
              <DetectionOverlay
                detections={detections} // 실시간 탐지 결과
                screenSize={screenSize}
                format={format}
                detectionResults={[]} // API 결과 표시 안 함
             />
           ) : null} ★★★ 조건부 렌더링 수정 ★★★ */}


           {/*
              API 분석 결과 오버레이 (나중에 구현할 부분 - AnalyzedResultOverlay)
              analyzedResults state는 useAnalysisApiHandler 훅에서 가져옵니다.
              분석 완료 상태일 때만 보여줍니다.
           */}
           {/* analyzedResults가 null이 아니고(분석이 한 번이라도 완료 또는 실패) 배열이고 길이가 0보다 클 때 */}
           {/* ★★★ analyzedResults가 null이 아닐 때 안전하게 접근하도록 조건 수정 ★★★ */}
           {analysisFinished && analyzedResults && analyzedResults.length > 0 ? ( // 분석 완료 상태이고 결과가 있을 때
               // ★★★ AnalyzedResultOverlay 컴포넌트 렌더링 (만들어야 함) ★★★
               // analyzedResults와 screenSize 정보를 prop으로 전달합니다.
               <AnalyzedResultOverlay results={analyzedResults} screenSize={screenSize} />
           ) : null} {/* ★★★ 조건부 렌더링 수정 ★★★ */}

            {/* API 분석 완료 후 결과는 없지만 카메라는 정지 상태인 경우 (예: 빈 화면에 분석 버튼 누름) */}
            {analysisFinished && analyzedResults !== null && analyzedResults.length === 0 ? (
                 <View style={styles.noDetectionMessage}><Text style={styles.noDetectionText}>객체 인식 결과 없음</Text></View>
            ) : null} {/* ★★★ 조건부 렌더링 수정 ★★★ */}


         </View> {/* 캡쳐 대상 영역 View 끝 */}
       </ViewShot> {/* ViewShot 끝 */}

        {/* ViewShot 영역 외에 표시될 UI 요소들 (절대 위치 사용) */}

        {/* 사과 또는 도넛 탐지 시 캡쳐 버튼 표시 */}
        {/* isAnalyzing 중이 아니고 분석 완료 상태가 아닐 때만 버튼 표시 */}
        {appleOrDonutDetected && !isAnalyzing && analyzedResults === null ? ( // 훅에서 가져온 isAnalyzing 사용
           <View style={styles.captureButtonContainer}>
               {/* AppleButton 컴포넌트의 onPress에 캡쳐 및 분석 함수 연결 */}
               <AppleButton
                   detections={detections} // 실시간 detections 전달 (보이게/안 보이게)
                   onPress={handleCaptureAndAnalyze} // 버튼 클릭 시 캡쳐+분석 실행
                   // isAnalyzing={isAnalyzing} // 버튼 내부에서 로딩 상태 사용 시 전달
               />
           </View>
        ) : null} {/* ★★★ 조건부 렌더링 수정 ★★★ */}


        {/* 분석 중 인디케이터 표시 */}
        {isAnalyzing ? ( // 훅에서 가져온 isAnalyzing 사용
            <View style={styles.loadingOverlay}>
                <ActivityIndicator size="large" color="#ffffff" />
                <Text style={styles.loadingText}>분석 중...</Text>
            </View>
        ) : null} {/* ★★★ 조건부 렌더링 수정 ★★★ */}


        {/* 탐지된 객체가 없을 때 힌트 메시지 */}
        {/* detections.length === 0 이고, isAnalyzing 중이 아니고, 분석 완료 상태가 아닐 때 표시 */}
        {detections.length === 0 && !isAnalyzing && analyzedResults === null ? ( // 훅에서 가져온 isAnalyzing 사용
           <AppleHint /> // AppleHint 컴포넌트가 자체 스타일을 가질 것으로 예상
        ) : null} {/* ★★★ 조건부 렌더링 수정 ★★★ */}


         {/* analysisError 상태 표시 (필요시) */}
          {/* analysisError && !isAnalyzing ? ( // 훅에서 가져온 analysisError 사용
              <View style={styles.errorOverlay}>
                   <Text style={styles.errorText}>Error: {analysisError}</Text>
              </View>
          ) : null */} {/* ★★★ 조건부 렌더링 수정 ★★★ */}


         {/* ★★★ 분석 완료 후 카메라를 다시 켜기 위한 버튼 등 UI 추가 필요 ★★★ */}
         {/* 분석 완료 상태일 때만 "다시 시작" 버튼 표시 */}
         {analysisFinished ? ( // 분석 완료 상태일 때
              <View style={styles.resumeButtonContainer}>
                   {/* 이 버튼을 누르면 analyzedResults와 analysisError를 null로 초기화 */}
                   {/* useAnalysisApiHandler 훅에 resetAnalysis 함수를 추가하고 임포트했다고 가정 */}
                   <Button title="다시 시작" onPress={() => {
                       // ★★★ useAnalysisApiHandler 훅에서 반환하는 resetAnalysis 함수 호출 ★★★
                       resetAnalysis();
                       // resetAnalysis 함수는 analyzedResults와 analysisError 상태를 null로 설정해야 합니다.
                       // isAnalyzing은 이미 false 상태일 것임.
                       // 이 상태 변경으로 인해 Camera의 isActive prop이 true로 바뀌면서 카메라가 다시 켜짐.
                   }} />
              </View>
         ) : null} {/* ★★★ 조건부 렌더링 수정 ★★★ */}


      </React.Fragment> 
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
     bottom: 100, // AppleButton이 자체 스타일로 관리한다면 이 스타일은 CameraView에서 제거
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
   noDetectionMessage: { // AppleHint 컴포넌트 대체
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
   // errorOverlay: { ... },
   // errorText: { ... },
    resumeButtonContainer: { // 다시 시작 버튼 컨테이너 스타일 추가
        position: 'absolute',
        bottom: 50,
        alignSelf: 'center',
        zIndex: 15, // 버튼이 로딩 오버레이 등 위에 표시되도록
    },
   // noResultOverlay: { ... }, // 결과 없을 때 메시지 컨테이너 (분석 완료 후)
   // noResultText: { ... }, // 결과 없을 때 메시지 텍스트 (분석 완료 후)
});