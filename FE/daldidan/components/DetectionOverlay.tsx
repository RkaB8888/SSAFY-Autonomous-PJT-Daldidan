import { Canvas, Group, Rect } from '@shopify/react-native-skia';
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { COCO_CLASS_NAMES } from '../constants/cocoClassNames';
import { Detection } from '../hooks/useObjectDetection';

interface Props {
  detections: Detection[];
  screenSize: { width: number; height: number };
  format: any;
}

export default function DetectionOverlay({
  detections,
  screenSize,
  format,
}: Props) {
  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Group>
          {detections.map((detection, i) => {
            const scaleX = screenSize.width / (format?.videoWidth || 1);
            const scaleY = screenSize.height / (format?.videoHeight || 1);

            const x = detection.x * scaleX;
            const y = detection.y * scaleY;
            const width = detection.width * scaleX;
            const height = detection.height * scaleY;

            return (
              <Group key={i}>
                <Rect
                  x={x}
                  y={y}
                  width={width}
                  height={height}
                  color='red'
                  style='stroke'
                  strokeWidth={3}
                />
              </Group>
            );
          })}
        </Group>
      </Canvas>
      {detections.map((detection, i) => {
        const scaleX = screenSize.width / (format?.videoWidth || 1);
        const scaleY = screenSize.height / (format?.videoHeight || 1);

        const x = detection.x * scaleX;
        const y = detection.y * scaleY;

        return (
          <View
            key={i}
            style={[
              styles.textContainer,
              {
                position: 'absolute',
                left: Math.max(0, Math.min(x, screenSize.width - 150)),
                top: Math.max(0, Math.min(y - 25, screenSize.height - 25)),
                width: 150,
              },
            ]}
          >
            <Text style={styles.text} numberOfLines={1}>
              {detection.class_id !== undefined &&
              detection.class_id < COCO_CLASS_NAMES.length
                ? COCO_CLASS_NAMES[detection.class_id]
                : 'unknown'}{' '}
              ({Math.round((detection.score ?? 0) * 100)}%)
            </Text>
          </View>
        );
      })}
    </>
  );
}

const styles = StyleSheet.create({
  textContainer: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 4,
    borderRadius: 4,
    zIndex: 1,
  },
  text: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});
