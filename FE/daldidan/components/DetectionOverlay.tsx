// components/DetectionOverlay.tsx

import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { Canvas, Group, Rect } from '@shopify/react-native-skia';
import { Detection } from '../hooks/useObjectDetection';

interface Props {
  detections: Detection[];
}

export default function DetectionOverlay({ detections }: Props) {
  return (
    <>
      {/* Skia Canvas 에 박스 그리기 */}
      <Canvas style={StyleSheet.absoluteFill}>
        <Group>
          {detections.map((d, i) => (
            <Rect
              key={i}
              x={d.x}
              y={d.y}
              width={d.width}
              height={d.height}
              style="stroke"
              strokeWidth={3}
              color="red"
            />
          ))}
        </Group>
      </Canvas>

      {/* 점수 텍스트 표시 */}
      {detections.map((d, i) => (
        <View
          key={i}
          style={[
            styles.label,
            {
              left: d.x,
              top: Math.max(0, d.y - 20),
            },
          ]}
        >
          <Text style={styles.text}>
            사과 {Math.round((d.score ?? 0) * 100)}%
          </Text>
        </View>
      ))}
    </>
  );
}

const styles = StyleSheet.create({
  label: {
    position: 'absolute',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 4,
    paddingVertical: 2,
    borderRadius: 4,
  },
  text: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});
