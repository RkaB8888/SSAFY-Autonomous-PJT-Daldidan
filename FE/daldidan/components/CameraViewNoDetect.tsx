import React, { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import { useObjectDetection } from '../hooks/useObjectDetection';
import DetectionOverlay from './DetectionOverlay';

export default function CameraView() {
  const device = useCameraDevice('back');
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });

  // 60fps 포맷 찾기
  const format =
    device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats[0];
  const fps = format ? Math.min(60, format.maxFps) : 30;

  const { hasPermission, detections, frameProcessor } =
    useObjectDetection(format);

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
      <Camera
        style={[
          StyleSheet.absoluteFill,
          detections.length === 0 && styles.grayedCamera,
        ]}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        fps={fps}
        format={format}
      />
      {detections.length === 0 ? (
        <View style={styles.noDetectionContainer}>
          <Text style={styles.noDetectionText}>사과 객체 인식되지 않음</Text>
        </View>
      ) : (
        <DetectionOverlay
          detections={detections}
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
