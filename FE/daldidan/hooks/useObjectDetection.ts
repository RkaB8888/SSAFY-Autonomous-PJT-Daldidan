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

export function useObjectDetection(format: any) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const cameraRef = useRef<Camera>(null);
  const frameCount = Worklets.createSharedValue(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>(
    []
  );
  const [hasPermission, setHasPermission] = useState(false);

  const { preprocessFrame, extractCroppedData, clamp, logWorklet } =
    useImageProcessing();

  // const updateDetections = (data: Detection[]) => {
  //   setDetections(data);
  // };

  // const updateDetectionsWorklet = Worklets.createRunOnJS(updateDetections);

  const updateDetectionsWorklet = useRef(
    Worklets.createRunOnJS((data: Detection[]) => {
      setDetections(data);
    })
  ).current;

  function nonMaxSuppression(
    boxes: { x1: number; y1: number; x2: number; y2: number }[],
    scores: number[],
    iouThreshold = 1
  ) {
    'worklet';
    // 점수 내림차순 인덱스
    const idxs = boxes.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const keep: number[] = [];

    for (let _i = 0; _i < idxs.length; _i++) {
      const i = idxs[_i];
      const bi = boxes[i];
      let shouldKeep = true;
      for (let j = 0; j < keep.length; j++) {
        const bj = boxes[keep[j]];
        // IoU 계산
        const xx1 = Math.max(bi.x1, bj.x1);
        const yy1 = Math.max(bi.y1, bj.y1);
        const xx2 = Math.min(bi.x2, bj.x2);
        const yy2 = Math.min(bi.y2, bj.y2);
        const w = Math.max(0, xx2 - xx1);
        const h = Math.max(0, yy2 - yy1);
        const inter = w * h;
        const areaI = (bi.x2 - bi.x1) * (bi.y2 - bi.y1);
        const areaJ = (bj.x2 - bj.x1) * (bj.y2 - bj.y1);
        const iou = inter / (areaI + areaJ - inter);
        if (iou > iouThreshold) {
          shouldKeep = false;
          break;
        }
      }
      if (shouldKeep) keep.push(i);
    }
    return keep;
  }

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

      const filteredBoxes = [];
      const filteredScores = [];
      const originalIndices = [];

      // ✅ 사과(class_id = 52) + score 조건 사전 필터링
      for (let i = 0; i < totalDetections; i++) {
        const classId = Math.round(classes[i]);
        const score = scores[i];
        if (classId !== 52 || score < CONFIDENCE_THRESHOLD) continue;

        const y1 = boxes[i * 4];
        const x1 = boxes[i * 4 + 1];
        const y2 = boxes[i * 4 + 2];
        const x2 = boxes[i * 4 + 3];

        filteredBoxes.push({ x1, y1, x2, y2 });
        filteredScores.push(score);
        originalIndices.push(i);
      }

      const keepIdx = nonMaxSuppression(filteredBoxes, filteredScores, 0.4);

      const detected: Detection[] = [];

      for (const idx of keepIdx) {
        const i = originalIndices[idx]; // 실제 index 복원
        const [y1, x1, y2, x2] = [
          boxes[i * 4],
          boxes[i * 4 + 1],
          boxes[i * 4 + 2],
          boxes[i * 4 + 3],
        ];

        if ([x1, y1, x2, y2].some((v) => isNaN(v) || v < 0 || v > 1)) continue;

        const inputSize = 320;
        const cropSize = 1080;
        const cropOffsetX = (1920 - cropSize) / 2;

        const x = clamp(x1 * cropSize + cropOffsetX, 0, frame.width);
        const y = clamp(y1 * cropSize, 0, frame.height);
        const width = clamp((x2 - x1) * cropSize, 0, frame.width - x);
        const height = clamp((y2 - y1) * cropSize, 0, frame.height - y);

        detected.push({
          x,
          y,
          width,
          height,
          score: filteredScores[idx], // 이미 저장된 값 재사용
          class_id: 52,
        });
      }

      return detected;
    } catch (error) {
      logWorklet(`[Worklet] Detection processing error: ${error}`);
      return [];
    }
  };

  function getGridKey(detection: Detection) {
    const grid = 80; // 80픽셀 단위로 확대
    return [
      detection.class_id,
      Math.round((detection.x + detection.width / 2) / grid) * grid,
      Math.round((detection.y + detection.height / 2) / grid) * grid,
    ].join('_');
  }

  const recentRequests = new Map();
  const DUPLICATE_TIMEOUT = 5000; // 5초

  // const capturePhoto = async (detection: Detection): Promise<string | null> => {
  //   ...
  // };

  const processExtractedData = useRef(async (extractedDataArray: any[]) => {
    const now = Date.now();
    console.log('[JS] Processing extracted data:', {
      totalItems: extractedDataArray.length,
      timestamp: now,
    });

    const processedKeys = new Set(
      detectionResults.map((r) => getGridKey(r.detection))
    );
    const uniqueItems = extractedDataArray.filter((item) => {
      const { detection } = item;
      const uniqueKey = getGridKey(detection);
      // 이미 처리된 객체는 건너뜀
      if (processedKeys.has(uniqueKey)) {
        console.log('[JS] Already processed detection, skipping:', uniqueKey);
        return false;
      }
      if (
        recentRequests.has(uniqueKey) &&
        now - recentRequests.get(uniqueKey) < DUPLICATE_TIMEOUT
      ) {
        console.log('[JS] Skipping duplicate detection:', {
          key: uniqueKey,
          timeSinceLastRequest: now - recentRequests.get(uniqueKey),
        });
        return false;
      }
      recentRequests.set(uniqueKey, now);
      return true;
    });

    console.log('[JS] Unique items to process:', {
      count: uniqueItems.length,
    });

    if (uniqueItems.length === 0) return;

    try {
      // const promises = uniqueItems.map(async (item) => {
      //   ...
      //   return null;
      // });
      // const results = await Promise.allSettled(promises);
      // const validResults = results
      //   .filter(...)
      //   .map(...);
      // if (validResults.length > 0) {
      //   setDetectionResults(...);
      // }
    } catch (error) {
      console.error('[JS] Batch processing error:', error);
    }
  }).current;

  // const runOnJSThread = Worklets.createRunOnJS(processExtractedData);
  // processExtractedData를 한 번만 래핑한 뒤, 콜백도 한 번만 생성
  const runOnJSThread = useRef(
    Worklets.createRunOnJS(processExtractedData)
  ).current;
  const SAMPLE_RATE = 28;

  const frameProcessor = useFrameProcessor(
    async (frame) => {
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
          const items = detections.map((detection) => ({
            detection: { ...detection },
            timestamp: Date.now(),
          }));

          runOnJSThread(items);
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
          require('../assets/1.tflite'),
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

  // detectionResults가 변경될 때마다 detections 업데이트
  useEffect(() => {
    if (detectionResults.length > 0) {
      const latestResult = detectionResults[detectionResults.length - 1];
      setDetections((prev) => {
        const updatedDetections = [...prev];
        const index = updatedDetections.findIndex(
          (d) =>
            d.class_id === latestResult.detection.class_id &&
            Math.abs(d.x - latestResult.detection.x) < 10 &&
            Math.abs(d.y - latestResult.detection.y) < 10
        );
        if (index !== -1) {
          updatedDetections[index] = {
            ...updatedDetections[index],
            sugar_content: latestResult.detection.sugar_content,
          };
        }
        return updatedDetections;
      });
    }
  }, [detectionResults]);

  return {
    hasPermission,
    detections,
    detectionResults,
    frameProcessor,
    cameraRef,
  };
}
