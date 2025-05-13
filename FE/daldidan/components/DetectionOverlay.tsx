import { Canvas, Group, Rect } from '@shopify/react-native-skia';
import React from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import { Detection } from '../hooks/types/objectDetection';
import { COCO_CLASS_NAMES } from '../constants/cocoClassNames';

interface Props {
  detections: Detection[];
  screenSize: { width: number; height: number };
  format: any;
  detectionResults: import('../hooks/types/objectDetection').DetectionResult[];
}

export default function DetectionOverlay({
  detections,
  detectionResults,
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

        // class_id만 같으면 표시
        const matched = detectionResults.find(
          (r) =>
            r.detection.class_id === detection.class_id &&
            r.detection.sugar_content !== undefined
        );

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
              {`${COCO_CLASS_NAMES[detection.class_id ?? 0] || 'Unknown'}${
                matched ? ` - 당도: ${matched.detection.sugar_content}Bx` : ''
              }`}
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
