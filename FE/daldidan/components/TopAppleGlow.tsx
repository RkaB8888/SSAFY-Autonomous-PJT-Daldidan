// components/TopAppleGlow.tsx
import React, { useEffect, useState } from 'react';
import { Group, Circle, Paint, BlurMask } from '@shopify/react-native-skia';
import { useSharedValue, withTiming, Easing } from 'react-native-reanimated';

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

export default function TopAppleGlow({ bbox, originalSize, screenSize }: Props) {
  const { width: originalWidth, height: originalHeight } = originalSize;
  const { width: screenWidth, height: screenHeight } = screenSize;

  // 1. bbox 좌표 변환
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
  const cy = (y1 + y2) / 2;
  const baseRadius = Math.max(x2 - x1, y2 - y1) * 0.7;

  // 2. 애니메이션 값 (state 사용)
  const radius = useSharedValue(baseRadius);
  const opacity = useSharedValue(1);

  const [finalRadius, setFinalRadius] = useState(baseRadius);
  const [finalOpacity, setFinalOpacity] = useState(1);

  useEffect(() => {
    radius.value = withTiming(baseRadius * 2.2, {
      duration: 800,
      easing: Easing.out(Easing.exp),
    });
    opacity.value = withTiming(0, {
      duration: 800,
      easing: Easing.in(Easing.ease),
    });

    // 폴백용 상태값 수동 동기화
    const interval = setInterval(() => {
      setFinalRadius(radius.value);
      setFinalOpacity(opacity.value);
    }, 16); // ~60fps

    setTimeout(() => clearInterval(interval), 900); // 정리

    return () => clearInterval(interval);
  }, []);

  return (
    <Group opacity={finalOpacity}>
      <Paint>
        <BlurMask blur={20} style="normal" />
      </Paint>
      <Circle cx={cx} cy={cy} r={finalRadius} color="rgba(255, 215, 0, 0.6)" />
    </Group>
  );
}
