import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import { Camera, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { CONFIDENCE_THRESHOLD, MODEL_INPUT_SIZE } from '../constants/model';
import { Detection, DetectionResult } from './types/objectDetection';
import { useImageProcessing } from './useImageProcessing';
import { useObjectAnalysis } from './useObjectAnalysis';

export function useObjectDetection(format: any) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const frameCount = Worklets.createSharedValue(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>(
    []
  );
  const [hasPermission, setHasPermission] = useState(false);

  const { preprocessFrame, extractCroppedData, clamp, logWorklet } =
    useImageProcessing();
  const { processImageData } = useObjectAnalysis();

  const updateDetections = (data: Detection[]) => {
    setDetections(data);
  };

  const updateDetectionsWorklet = Worklets.createRunOnJS(updateDetections);

  const processDetectionsInWorklet = (frame: any, model: TensorflowModel) => {
    'worklet';
    try {
      const resized = preprocessFrame(frame, MODEL_INPUT_SIZE);
      const outputs = model.runSync([resized]);
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
        const score = scores[i];
        if (score < CONFIDENCE_THRESHOLD) continue;

        const [y1, x1, y2, x2] = [
          boxes[i * 4],
          boxes[i * 4 + 1],
          boxes[i * 4 + 2],
          boxes[i * 4 + 3],
        ];

        if ([x1, y1, x2, y2].some((v) => isNaN(v) || v < 0 || v > 1)) continue;

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

      return detected;
    } catch (error) {
      logWorklet(`[Worklet] Detection processing error: ${error}`);
      return [];
    }
  };

  const processExtractedData = async (extractedDataArray: any[]) => {
    for (const item of extractedDataArray) {
      const { detection, croppedData, timestamp } = item;
      if (!croppedData) {
        console.warn('[JS] No cropped data received');
        continue;
      }
      try {
        if (!croppedData.data || !Array.isArray(croppedData.data)) {
          console.warn('[JS] Invalid data structure received:', croppedData);
          continue;
        }

        const uint8Array = new Uint8Array(croppedData.data);
        if (uint8Array.length === 0) {
          console.warn('[JS] Empty data received');
          continue;
        }

        const result = await processImageData(uint8Array, detection, timestamp);
        if (result) {
          setDetectionResults((prev) => [...prev, result]);
        }
      } catch (error) {
        console.error('[JS] API request error:', error);
      }
    }
  };

  const runOnJSThread = Worklets.createRunOnJS(processExtractedData);

  const SAMPLE_RATE = 10;

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!modelRef.current) {
        logWorklet('[Worklet] Model not loaded');
        return;
      }

      frameCount.value = (frameCount.value + 1) % SAMPLE_RATE;
      if (frameCount.value !== 0) return;

      try {
        logWorklet('[Worklet] Starting frame processing');

        const detections = processDetectionsInWorklet(frame, modelRef.current);

        if (detections && detections.length > 0) {
          const extractedData = detections.map((detection) => {
            const cropped = extractCroppedData(frame, detection);
            return {
              detection: { ...detection },
              croppedData: cropped,
              timestamp: Date.now(),
            };
          });

          runOnJSThread(extractedData);
          updateDetectionsWorklet(detections);
        } else {
          updateDetectionsWorklet([]);
        }
      } catch (error) {
        logWorklet(`[Worklet] Frame processing error: ${error}`);
        updateDetectionsWorklet([]);
      }
    },
    [updateDetectionsWorklet, runOnJSThread, logWorklet]
  );

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

    return () => {
      modelRef.current = null;
      frameCount.value = 0;
      setDetections([]);
      setDetectionResults([]);
    };
  }, []);

  return {
    hasPermission,
    detections,
    detectionResults,
    frameProcessor,
  };
}
