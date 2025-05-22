import React, { useRef, useEffect, useState } from 'react';
import { StyleSheet, Animated, View } from 'react-native';
import { Detection } from '../hooks/types/objectDetection';

interface Props {
  detections: Detection[];
  screenSize: { width: number; height: number };
  format: any;
}

export default function DetectionOverlay({ detections, screenSize }: Props) {
  const prevBoxes = useRef<Detection[]>([]);
  const [smoothed, setSmoothed] = useState<Detection[]>([]);
  const glowAnim = useRef(new Animated.Value(0)).current;

  const SMOOTH_FACTOR = 0.2;

  // 위치 스무딩
  useEffect(() => {
    if (detections.length !== prevBoxes.current.length) {
      prevBoxes.current = detections.map((d) => ({ ...d }));
      setSmoothed(detections);
      return;
    }

      const next = detections.map((d, i) => {
      const prev = prevBoxes.current[i];
      return {
        ...d,
        x: prev.x + (d.x - prev.x) * SMOOTH_FACTOR,
        y: prev.y + (d.y - prev.y) * SMOOTH_FACTOR,
        width: prev.width + (d.width - prev.width) * SMOOTH_FACTOR,
        height: prev.height + (d.height - prev.height) * SMOOTH_FACTOR,
      };
    });

    prevBoxes.current = next;
    setSmoothed(next);
  }, [detections]);

  // glow 애니메이션
  useEffect(() => {
    if (detections.some((d) => d.class_id === 52)) {
      glowAnim.setValue(0);
      Animated.timing(glowAnim, {
        toValue: 1,
        duration: 1200,
        useNativeDriver: false,
      }).start();
    }
  }, [detections]);

  return (
    <>
      {smoothed.map((detection, i) => {
        if (detection.class_id !== 52) return null;

        // 카메라 프레임 -> 화면 좌표 변환
        const frameW = 1920;
        const frameH = 1080;
        const screenW = screenSize.width;
        const screenH = screenSize.height;

        const rotated = {
          x: detection.y,
          y: frameW - detection.x - detection.width,
          width: detection.height,
          height: detection.width,
        };

        const scaleX = screenW / frameH;
        const scaleY = screenH / frameW;

        const x = rotated.x * scaleX;
        const y = rotated.y * scaleY;
        const width = rotated.width * scaleX;
        const height = rotated.height * scaleY;

        const glowSize = Math.max(width, height) * 1.5;

        return (
          <Animated.View
            key={`glow-${i}`}
            style={{
              position: 'absolute',
              left: x + width / 2 - glowSize / 2,
              top: y + height / 2 - glowSize / 2,
              width: glowSize,
              height: glowSize,
              borderRadius: glowSize / 2,
              backgroundColor: 'rgba(233, 172, 103, 0.54)',
              transform: [
                {
                  scale: glowAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0.6, 1.2],
                  }),
                },
              ],
              opacity: glowAnim.interpolate({
                inputRange: [0, 1],
                outputRange: [0.7, 0],
              }),
              zIndex: 5,
            }}
          />
        );
      })}
    </>
  );
}

const styles = StyleSheet.create({});
