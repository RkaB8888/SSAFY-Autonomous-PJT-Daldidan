import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import { Camera, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { CONFIDENCE_THRESHOLD, MODEL_INPUT_SIZE } from '../constants/model';

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  score?: number;
  class_id?: number;
}

export function useObjectDetection(format: any) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const frameCount = Worklets.createSharedValue(0);
  const lastDetectionsRef = useRef<Detection[]>([]);
  const { resize } = useResizePlugin();
  const [detections, setDetections] = useState<Detection[]>([]);
  const [hasPermission, setHasPermission] = useState(false);

  // 입력 전처리 함수
  const preprocessFrame = (frame: any, targetSize: number) => {
    'worklet';
    return resize(frame, {
      scale: { width: targetSize, height: targetSize },
      pixelFormat: 'rgb',
      dataType: 'uint8',
    });
  };

  // 감지 결과 처리 함수
  const updateDetections = (data: Detection[]) => {
    setDetections(data);
  };

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
        console.log('Attempting to load model...');
        const model = await loadTensorflowModel(
          require('../assets/model.tflite'),
          'gpu' as TensorflowModelDelegate
        );
        console.log('Model loaded successfully');
        modelRef.current = model;
      } catch (error: any) {
        console.error('Model loading error:', error);
        Alert.alert('Model Error', error.message);
      }
    };
    loadModel();
  }, []);

  // 상수 조절로 샘플링 빈도 변경 (3 → 2프레임마다 1회 등)
  const SAMPLE_RATE = 6; // 1초에 10번 연산

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!modelRef.current) return;

      // 프레임 카운트 업데이트 (Worklet 내부에서 값 직접 수정)
      frameCount.value = (frameCount.value + 1) % SAMPLE_RATE;
      if (frameCount.value !== 0) return;

      try {
        const resized = preprocessFrame(frame, MODEL_INPUT_SIZE);
        const outputs = modelRef.current.runSync([resized]);
        const boxes = outputs[0] as Float32Array;
        const classes = outputs[1] as Float32Array;
        const scores = outputs[2] as Float32Array;
        const numDetections = outputs[3] as Float32Array;

        const totalDetections = Math.min(
          Math.round(numDetections[0] || 0),
          scores.length
        );

        const detected: Detection[] = [];

        // 탐지 결과 후처리 최적화
        for (let i = 0; i < totalDetections; i++) {
          const score = scores[i];
          if (score < CONFIDENCE_THRESHOLD) continue;

          // 박스 좌표 검증 통합
          const [y1, x1, y2, x2] = [
            boxes[i * 4],
            boxes[i * 4 + 1],
            boxes[i * 4 + 2],
            boxes[i * 4 + 3],
          ];

          if ([x1, y1, x2, y2].some((v) => isNaN(v) || v < 0 || v > 1))
            continue;

          // 실제 좌표 변환 (Math.min/max 제거 → clamp 함수로 대체)
          const clamp = (value: number, min: number, max: number) =>
            Math.max(min, Math.min(value, max));

          const x = clamp(x1 * frame.width, 0, frame.width);
          const y = clamp(y1 * frame.height, 0, frame.height);
          const width = clamp((x2 - x1) * frame.width, 0, frame.width - x);
          const height = clamp((y2 - y1) * frame.height, 0, frame.height - y);

          detected.push({
            x,
            y,
            width,
            height,
            score,
            class_id: Math.round(classes[i]),
          });
        }

        // 상태 업데이트 최소화
        if (detected.length > 0 || lastDetectionsRef.current.length > 0) {
          updateDetectionsWorklet(detected);
          lastDetectionsRef.current = detected;
        }
      } catch (error) {
        console.error('Frame processing error:', error);
      }
    },
    [updateDetectionsWorklet]
  );

  return { hasPermission, detections, frameProcessor };
}
