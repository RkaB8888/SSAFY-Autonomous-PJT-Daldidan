import { Canvas, Group, Rect } from '@shopify/react-native-skia';
import React, { useEffect, useRef, useState } from 'react';
import { Alert, StyleSheet, Text, View } from 'react-native';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';

// 🔥 COCO 클래스 이름 배열 추가
const COCO_CLASS_NAMES = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'TV',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];

// 모델 설정
const MODEL_INPUT_SIZE = 320;
const CONFIDENCE_THRESHOLD = 0.3;

interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  score?: number;
  class_id?: number;
}

export default function App() {
  const device = useCameraDevice('back');
  const [hasPermission, setHasPermission] = useState(false);
  const modelRef = useRef<TensorflowModel | null>(null);
  const frameCount = Worklets.createSharedValue(0);
  const { resize } = useResizePlugin();
  const [detections, setDetections] = useState<Detection[]>([]);
  const [screenSize, setScreenSize] = useState({ width: 0, height: 0 });

  // 입력 전처리 함수 추가
  const preprocessFrame = (frame, targetSize) => {
    'worklet';
    // EfficientDet 모델 요구사항에 맞게 전처리
    // 메타데이터: quantization: linear 0.0078125 * (q - 127)
    return resize(frame, {
      scale: { width: targetSize, height: targetSize },
      pixelFormat: 'rgb',
      dataType: 'uint8',
      normalization: {
        mean: [127, 127, 127], // zero point = 127
        std: [128, 128, 128], // 1/0.0078125 = 128
      },
    });
  };

  // 60fps 포맷 찾기
  const format =
    device?.formats.find((f) => f.maxFps >= 60) ?? device?.formats[0];
  const fps = format ? Math.min(60, format.maxFps) : 30;

  // 감지 결과 처리 함수
  const updateDetections = (data: Detection[]) => {
    setDetections(data);
  };

  // Worklet 함수 생성
  const updateDetectionsWorklet = Worklets.createRunOnJS(updateDetections);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const model = await loadTensorflowModel(
          require('../../assets/model.tflite'),
          'gpu' as TensorflowModelDelegate
        );
        console.log('Model loaded successfully:', model);
        modelRef.current = model;
      } catch (error: any) {
        console.error('Model loading error:', error);
        Alert.alert('Model Error', error.message);
      }
    };
    loadModel();
  }, []);

  // frameProcessor 함수 내부 수정
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!modelRef.current) return;

      // 프레임 카운터 업데이트 (성능 최적화)
      frameCount.value = (frameCount.value + 1) % 1; // 10프레임마다 처리로 변경
      if (frameCount.value !== 0) return;

      try {
        console.log('프레임 처리 시작');

        // 최적화된 전처리 함수 사용
        const resized = preprocessFrame(frame, MODEL_INPUT_SIZE);
        console.log('이미지 리사이즈 완료');

        const outputs = modelRef.current.runSync([resized]);
        console.log('모델 추론 완료');

        // 올바른 출력 텐서 순서로 접근 (EfficientDet 모델 기준)
        const boxes = outputs[0] as Float32Array; // 바운딩 박스 (StatefulPartitionedCall:0)
        const classes = outputs[1] as Float32Array; // 클래스 ID (StatefulPartitionedCall:1)
        const scores = outputs[2] as Float32Array; // 신뢰도 점수 (StatefulPartitionedCall:2)
        const numDetections = outputs[3] as Float32Array; // 검출 수 (StatefulPartitionedCall:3)

        // 실제 감지된 객체 수 사용 (최대 25개)
        const totalDetections = Math.min(
          Math.round(numDetections[0] || 0), // 실제 감지된 객체 수 또는 0
          scores.length
        );

        console.log('감지된 객체 수:', totalDetections);

        const detected: Detection[] = [];

        for (let i = 0; i < totalDetections; i++) {
          // 신뢰도 임계값 확인
          if (scores[i] < CONFIDENCE_THRESHOLD) continue;

          // EfficientDet 출력 형식: [y1, x1, y2, x2] (정규화된 좌표)
          const y1 = boxes[i * 4];
          const x1 = boxes[i * 4 + 1];
          const y2 = boxes[i * 4 + 2];
          const x2 = boxes[i * 4 + 3];

          // 유효한 좌표값 검증
          if ([x1, y1, x2, y2].some((v) => isNaN(v) || v < 0 || v > 1)) {
            console.log(`detection[${i}] 유효하지 않은 좌표, 건너뜀`);
            continue;
          }

          // 화면 크기에 맞게 변환 (좌표 순서 수정)
          const x = Math.min(Math.max(x1 * frame.width, 0), frame.width);
          const y = Math.min(Math.max(y1 * frame.height, 0), frame.height);
          const width = Math.min((x2 - x1) * frame.width, frame.width - x);
          const height = Math.min((y2 - y1) * frame.height, frame.height - y);

          const classId = Math.round(classes[i]);
          const className =
            classId < COCO_CLASS_NAMES.length
              ? COCO_CLASS_NAMES[classId]
              : 'unknown';

          console.log(
            `감지된 객체[${i}]: ${className} (신뢰도: ${Math.round(
              scores[i] * 100
            )}%) ` +
              `위치: x=${Math.round(x)}, y=${Math.round(y)}, w=${Math.round(
                width
              )}, h=${Math.round(height)}`
          );

          detected.push({
            x,
            y,
            width,
            height,
            score: scores[i],
            class_id: classId,
          });
        }

        console.log(`총 ${detected.length}개의 객체 감지 완료`);
        updateDetectionsWorklet(detected);
      } catch (error) {
        console.error('객체 감지 중 오류 발생:', error);
      }
    },
    [updateDetectionsWorklet]
  );
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
      <Canvas style={StyleSheet.absoluteFill}>
        <Group>
          {detections.map((detection, i) => {
            // 화면 비율에 맞게 좌표 조정
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
        // 화면 비율에 맞게 좌표 조정
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
              ({Math.round(detection.score! * 100)}%)
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
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
