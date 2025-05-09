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

  // frameProcessor 함수
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!modelRef.current) return;

      frameCount.value = (frameCount.value + 1) % 1;
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

        for (let i = 0; i < totalDetections; i++) {
          if (scores[i] < CONFIDENCE_THRESHOLD) continue;

          const y1 = boxes[i * 4];
          const x1 = boxes[i * 4 + 1];
          const y2 = boxes[i * 4 + 2];
          const x2 = boxes[i * 4 + 3];

          if ([x1, y1, x2, y2].some((v) => isNaN(v) || v < 0 || v > 1))
            continue;

          const x = Math.min(Math.max(x1 * frame.width, 0), frame.width);
          const y = Math.min(Math.max(y1 * frame.height, 0), frame.height);
          const width = Math.min((x2 - x1) * frame.width, frame.width - x);
          const height = Math.min((y2 - y1) * frame.height, frame.height - y);

          const classId = Math.round(classes[i]);

          detected.push({
            x,
            y,
            width,
            height,
            score: scores[i],
            class_id: classId,
          });
        }
        updateDetectionsWorklet(detected);
      } catch (error) {
        console.error('Frame processing error:', error);
      }
    },
    [updateDetectionsWorklet]
  );

  return { hasPermission, detections, frameProcessor };
}
