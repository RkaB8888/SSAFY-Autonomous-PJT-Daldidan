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

            // 객체 크기에 따라 선 두께 조절
            const strokeWidth = Math.max(
              2,
              Math.min(4, Math.min(width, height) * 0.02)
            );

            return (
              <Group key={i}>
                {/* 외부 박스 */}
                <Rect
                  x={x}
                  y={y}
                  width={width}
                  height={height}
                  color='rgba(255, 0, 0, 0.8)'
                  style='stroke'
                  strokeWidth={strokeWidth}
                />
                {/* 모서리 강조 */}
                <Group>
                  {/* 좌상단 모서리 */}
                  <Rect
                    x={x}
                    y={y}
                    width={width * 0.2}
                    height={strokeWidth * 2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  <Rect
                    x={x}
                    y={y}
                    width={strokeWidth * 2}
                    height={height * 0.2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  {/* 우상단 모서리 */}
                  <Rect
                    x={x + width - width * 0.2}
                    y={y}
                    width={width * 0.2}
                    height={strokeWidth * 2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  <Rect
                    x={x + width - strokeWidth * 2}
                    y={y}
                    width={strokeWidth * 2}
                    height={height * 0.2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  {/* 좌하단 모서리 */}
                  <Rect
                    x={x}
                    y={y + height - height * 0.2}
                    width={width * 0.2}
                    height={strokeWidth * 2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  <Rect
                    x={x}
                    y={y + height - height * 0.2}
                    width={strokeWidth * 2}
                    height={height * 0.2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  {/* 우하단 모서리 */}
                  <Rect
                    x={x + width - width * 0.2}
                    y={y + height - height * 0.2}
                    width={width * 0.2}
                    height={strokeWidth * 2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                  <Rect
                    x={x + width - strokeWidth * 2}
                    y={y + height - height * 0.2}
                    width={strokeWidth * 2}
                    height={height * 0.2}
                    color='rgba(255, 0, 0, 0.8)'
                  />
                </Group>
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
        const width = detection.width * scaleX;
        const height = detection.height * scaleY;

        // 객체 크기에 따라 텍스트 크기 조절
        const fontSize = Math.max(
          10,
          Math.min(14, Math.min(width, height) * 0.1)
        );

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
                backgroundColor: 'rgba(0, 0, 0, 0.7)',
                borderWidth: 1,
                borderColor: 'rgba(255, 0, 0, 0.8)',
              },
            ]}
          >
            <Text style={[styles.text, { fontSize }]} numberOfLines={1}>
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
    padding: 4,
    borderRadius: 4,
    zIndex: 1,
  },
  text: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
