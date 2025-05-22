// components/TopAppleCrown.tsx
import React from 'react';
import { StyleSheet, View, Platform } from 'react-native';
import LottieView from 'lottie-react-native';

interface Props {
  bbox: {
    xmin: number;
    ymin: number;
    xmax: number;
    ymax: number;
  };
  originalSize: {
    width: number;
    height: number;
  };
  screenSize: {
    width: number;
    height: number;
  };
}

export default function TopAppleCrown({ bbox, originalSize, screenSize }: Props) {
  const { width: originalWidth, height: originalHeight } = originalSize;
  const { width: screenWidth, height: screenHeight } = screenSize;

  // bbox 좌표 변환 (회전 포함)
  const rotatedX1 = originalHeight - bbox.ymax;
  const rotatedY1 = bbox.xmin;
  const rotatedX2 = originalHeight - bbox.ymin;
  const rotatedY2 = bbox.xmax;

  const rotatedImageWidth = originalHeight;
  const rotatedImageHeight = originalWidth;
  const scale = screenHeight / rotatedImageHeight;
  const offsetX = (screenWidth - rotatedImageWidth * scale) / 2;

  const x1 = rotatedX1 * scale + offsetX;
  const y1 = rotatedY1 * scale;
  const x2 = rotatedX2 * scale + offsetX;
  const y2 = rotatedY2 * scale;

  const cx = (x1 + x2) / 2;
  const top = Math.min(y1, y2);

  const crownWidth = Math.max(x2 - x1, y2 - y1) * 0.4;

  return (
    <View
      style={{
        position: 'absolute',
        left: cx - crownWidth / 2,
        top: top - crownWidth * 0.9,
        width: crownWidth,
        height: crownWidth,
        zIndex: 10,
        elevation: 0,
      }}
      pointerEvents="none"
      needsOffscreenAlphaCompositing={true} // 추가
    >
      <LottieView
        source={require('../assets/lottie/crown.json')}
        autoPlay
        loop={false}
        style={StyleSheet.absoluteFill}
        {...(Platform.OS === 'android' && { renderMode: 'SOFTWARE' })} // Android에서만 renderMode를 SOFTWARE로 설정
      />
    </View>
  );
} 