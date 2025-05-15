import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, AppState } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useObjectDetection } from '../hooks/useObjectDetection';
import DetectionOverlay from './DetectionOverlay';

export default function CameraView() {
  const device = useCameraDevice('back');
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });
  const [appState, setAppState] = useState('active');

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
          // style={[
          //   StyleSheet.absoluteFill,
          //   detections.length === 0 && styles.grayedCamera,
          // ]}
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={appState === 'active'}
          frameProcessor={frameProcessor}
          fps={fps}
          format={format}
          photo={true}
        />
      )}
      {detections.length === 0 ? (
        <View style={styles.noDetectionContainer}>
          <Text style={styles.noDetectionText}>사과 객체 인식되지 않음</Text>
        </View>
      ) : (
        <DetectionOverlay
          detections={detections}
          detectionResults={detectionResults}
          screenSize={screenSize}
          format={format}
        />
      )}
    </View>
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
