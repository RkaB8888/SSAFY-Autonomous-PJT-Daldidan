import React, { useState } from 'react';
import { StyleSheet, View } from 'react-native';
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
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        fps={fps}
        format={format}
      />
      <DetectionOverlay
        detections={detections}
        screenSize={screenSize}
        format={format}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
});
