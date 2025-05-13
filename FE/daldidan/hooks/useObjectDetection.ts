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
}

export function useObjectDetection(format: any) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const frameCount = Worklets.createSharedValue(0);
  const { resize } = useResizePlugin();
  const [detections, setDetections] = useState<Detection[]>([]);
  const [hasPermission, setHasPermission] = useState(false);

const preprocessFrame = (frame: any, targetSize: number) => {
  'worklet';

  const resized = resize(frame, {
    scale: { width: targetSize, height: targetSize },
    pixelFormat: 'rgb',
    dataType: 'float32',
    rotation: '90deg',
    // resizeMode: 'contain', // ← 핵심! letterbox (패딩 유지 리사이즈)
    // background: [0, 0, 0], // 검은 패딩
  });

  for (let i = 0; i < resized.length; i++) {
    resized[i] /= 255.0;
  }

    // RGB → BGR 변환 실험
  for (let i = 0; i < resized.length; i += 3) {
    const r = resized[i];
    const g = resized[i + 1];
    const b = resized[i + 2];
    resized[i] = b;
    resized[i + 1] = g;
    resized[i + 2] = r;
  }

  return resized;
};



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
        const model = await loadTensorflowModel(
          require('../assets/model3.tflite'),
          'gpu' as TensorflowModelDelegate
        );
        modelRef.current = model;
      } catch (error: any) {
        console.error('Model loading error:', error);
        Alert.alert('Model Error', error.message);
      }
    };
    loadModel();
  }, []);

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';

    if (!modelRef.current) return;

    frameCount.value = (frameCount.value + 1) % 3;
    if (frameCount.value !== 0) return;

    try {
      const resized = preprocessFrame(frame, MODEL_INPUT_SIZE);
      const outputs = modelRef.current.runSync([resized]);
      const output = outputs[0] as Float32Array; // [1, 5, 8400] flatten
      console.log('📦 output.length:', output.length);
      console.log('📦 sample:', Array.from(output.slice(0, 20)));
      console.log("📸 frame ID:", frame.timestamp); // VisionCamera 프레임마다 고유 ID
      console.log("🎨 frame hash sample:", resized[0], resized[10], resized[100]); // 입력 일부
      console.log("📷 입력 hash:", resized[0], resized[10], resized[100]);



      const numAnchors = 8400;
      const screenRatioX = frame.width / MODEL_INPUT_SIZE;
      const screenRatioY = frame.height / MODEL_INPUT_SIZE;

const detected: Detection[] = [];

for (let i = 0; i < 300; i++) {
  const offset = i * 6;
  const x_center = output[offset + 0];
  const y_center = output[offset + 1];
  const w = output[offset + 2];
  const h = output[offset + 3];
  const score = output[offset + 4];
  const classProb = output[offset + 5]; // class 확률 (1.0에 가까울수록 확실)

  console.log("score: ", score)


  if (score > CONFIDENCE_THRESHOLD) {
    const box = {
      x: (x_center - w / 2) * screenRatioX,
      y: (y_center - h / 2) * screenRatioY,
      width: w * screenRatioX,
      height: h * screenRatioY,
      score,
    };
    detected.push(box);

    // 📦 로그 찍기
    console.log(`🍎 사과 탐지됨 → [x: ${box.x.toFixed(1)}, y: ${box.y.toFixed(1)}, w: ${box.width.toFixed(1)}, h: ${box.height.toFixed(1)}, score: ${score.toFixed(3)}]`);
  }
}




      updateDetectionsWorklet(detected);
    } catch (error: any) {
      console.error('Frame processing error:', error);
    }
  }, [updateDetectionsWorklet]);

  return { hasPermission, detections, frameProcessor };
}
