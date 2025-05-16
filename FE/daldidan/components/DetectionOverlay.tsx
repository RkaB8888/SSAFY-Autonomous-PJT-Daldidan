import { Canvas, Group, Rect } from '@shopify/react-native-skia';
import React, { useRef, useEffect, useState } from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import { Detection } from '../hooks/types/objectDetection';
import { COCO_CLASS_NAMES } from '../constants/cocoClassNames';

interface Props {
  detections: Detection[];
  screenSize: { width: number; height: number };
  format: any;
  // detectionResults: import('../hooks/types/objectDetection').DetectionResult[];
}

export default function DetectionOverlay({
  detections,
  // detectionResults,
  screenSize,
  format,
}: Props) {
    const prevBoxes = useRef<Detection[]>([]);
      const [smoothed, setSmoothed] = useState<Detection[]>([]);

      const SMOOTH_FACTOR = 0.2; // 부드럽게 이동 (0.0~1.0)

      useEffect(() => {
        if (detections.length !== prevBoxes.current.length) {
          // 초기화 or 박스 개수가 바뀐 경우
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

  return (
    <>
      <Canvas style={StyleSheet.absoluteFill}>
        <Group>
          {smoothed.map((detection, i) => {
                // 📌 DetectionOverlay 내에서, detection -> 화면상 위치 변환
              const frameW = 1920;
              const frameH = 1080;
              const screenW = screenSize.width;
              const screenH = screenSize.height;

              // 90도 회전 + 위치 변환
              const rotated = {
                x: detection.y,
                y: frameW - detection.x - detection.width,
                width: detection.height,
                height: detection.width,
              };

              const scaleX = screenW / frameH; // frameH = 1080 → now width axis
              const scaleY = screenH / frameW; // frameW = 1920 → now height axis

              const x = rotated.x * scaleX;
              const y = rotated.y * scaleY;
              const width = rotated.width * scaleX;
              const height = rotated.height * scaleY;

              console.log(`[Debug ${i}]`);
              console.log('original:', detection);
              console.log('rotated:', rotated);
              console.log('screen:', {
                x: x.toFixed(1),
                y: y.toFixed(1),
                width: width.toFixed(1),
                height: height.toFixed(1),
              });

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
                  color='rgba(255, 255, 255, 0.8)'
                  style='stroke'
                  strokeWidth={strokeWidth}
                />
                
              </Group>
            );
          })}
        </Group>
      </Canvas>
      {smoothed.map((detection, i) => {
       // 📌 DetectionOverlay 내에서, detection -> 화면상 위치 변환
        const frameW = 1920;
        const frameH = 1080;
        const screenW = screenSize.width;
        const screenH = screenSize.height;

        // 90도 회전 + 위치 변환
        const rotated = {
          x: detection.y,
          y: frameW - detection.x - detection.width,
          width: detection.height,
          height: detection.width,
        };

        const scaleX = screenW / frameH; // frameH = 1080 → now width axis
        const scaleY = screenH / frameW; // frameW = 1920 → now height axis

        const x = rotated.x * scaleX;
        const y = rotated.y * scaleY;
        const width = rotated.width * scaleX;
        const height = rotated.height * scaleY;

        // 객체 크기에 따라 텍스트 크기 조절
        const fontSize = Math.max(
          10,
          Math.min(14, Math.min(width, height) * 0.1)
        );

        // const matched = detectionResults.find(
        //   (r) =>
        //     r.detection.class_id === detection.class_id &&
        //     r.detection.sugar_content !== undefined
        // );

        // return (
        //   <View
        //     key={i}
        //     style={[
        //       styles.textContainer,
        //       {
        //         position: 'absolute',
        //         left: Math.max(0, Math.min(x, screenSize.width - 150)),
        //         top: Math.max(0, Math.min(y - 25, screenSize.height - 25)),
        //         width: 150,
        //         backgroundColor: 'rgba(0, 0, 0, 0.7)',
        //         borderWidth: 1,
        //         borderColor: 'rgba(255, 0, 0, 0.8)',
        //       },
        //     ]}
        //   >
        //     <Text style={[styles.text, { fontSize }]} numberOfLines={1}>
        //       {/* {`${COCO_CLASS_NAMES[detection.class_id ?? 0] || 'Unknown'}${
        //         matched ? ` - 당도: ${matched.detection.sugar_content}Bx` : ''
        //       }`} */}
        //     </Text>
        //   </View>
        // );
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
