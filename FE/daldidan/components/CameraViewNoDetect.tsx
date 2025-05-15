import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, AppState } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useObjectDetection } from '../hooks/useObjectDetection';
import DetectionOverlay from './DetectionOverlay';
import AppleButton from './AppleButton';
import ViewShot, { captureRef } from 'react-native-view-shot';
import AppleHint from './AppleHint'
import AppleProcessing from './AppleProcessing';
import AppleBar from './AppleBar';

export default function CameraView() {
  const device = useCameraDevice('back');
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const [appState, setAppState] = useState('active');

  const viewShotRef = useRef(null);
  const [cameraPaused, setCameraPaused] = useState(false);
  const handleApplePress = async () => {
    if (cameraPaused) {
    // 🔁 다시 카메라 켜기
    console.log('[🎥 재시작]');
    setCameraPaused(false);
    setProcessingStage('none');
    return;
  }
      try {
        setCameraPaused(true); // 🔸 카메라 정지
        setProcessingStage('juicing');

        const uri = await captureRef(viewShotRef, {
          format: 'jpg',
          quality: 0.8,
        });

        console.log('[🍎 캡처 완료] 이미지 경로:', uri);
        setTimeout(() => {
          setProcessingStage('none');
          setCameraPaused(false);
          console.log('[🎬 애니메이션 종료, 카메라 재개]');
        }, 3000);

        // 🔜 다음: 애니메이션 표시 + 서버 전송
      } catch (err) {
        console.error('캡처 실패:', err);
        setCameraPaused(false);
      }
    };

    const [processingStage, setProcessingStage] = useState<'none' | 'juicing'>('none');
    const handlePause = () => {
      setProcessingStage('juicing');
    };

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (nextAppState) => {
      console.log('AppState changed:', nextAppState);
      setAppState(nextAppState);
    });
    return () => subscription.remove();
  }, []);

  const format =
    device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats[0];
  const fps = format ? Math.min(60, format.maxFps) : 30;

  const {
    hasPermission,
    detections,
    frameProcessor,
    cameraRef,
    detectionResults,
  } = useObjectDetection(format);

  if (!hasPermission || !device || !format) {
    return <View style={styles.container} />;
  }

  return (
    <ViewShot ref={viewShotRef} style={StyleSheet.absoluteFill} options={{ format: 'jpg', quality: 0.8 }}>
    <View
      style={StyleSheet.absoluteFill}
      onLayout={(event) => {
        const { width, height } = event.nativeEvent.layout;
        setScreenSize({ width, height });
      }}
    >
      {/* appState가 active일 때만 Camera를 마운트 */}
      {appState === 'active' && (
        <Camera
          ref={cameraRef}
          style={[
            StyleSheet.absoluteFill,
            detections.length === 0 && styles.grayedCamera,
          ]}
          device={device}
          isActive={appState === 'active' && !cameraPaused}
          frameProcessor={frameProcessor}
          fps={fps}
          format={format}
          photo={true}
        />
        
      )}
         <AppleBar detections={detections} />
      {detections.length === 0 ? (
        
        <AppleHint />
        // <View style={styles.noDetectionContainer}>
        //   <Text style={styles.noDetectionText}>🍎사과를 비춰주세요🍎</Text>
        // </View>
      ) : (
        <>
        {/* <DetectionOverlay
          detections={detections}
          detectionResults={detectionResults}
          screenSize={screenSize}
          format={format}
        /> */}
        <AppleButton
       detections={detections}
       onPress={handleApplePress}
    />
    </>
      )}
      {processingStage !== 'none' && (
      <AppleProcessing status={processingStage} />
    )}
    </View>
  </ViewShot>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  grayedCamera: {
    opacity: 0.7,
  },
  noDetectionContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  noDetectionText: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    padding: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: 10,
  },
});
