import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';
import {
  loadTensorflowModel,
  TensorflowModel,
  TensorflowModelDelegate,
} from 'react-native-fast-tflite';
import {
  Camera,
  useFrameProcessor,
  useCameraDevice,
} from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import * as ImageManipulator from 'expo-image-manipulator';
import { CONFIDENCE_THRESHOLD, MODEL_INPUT_SIZE } from '../constants/model';
import { COCO_CLASS_NAMES } from '../constants/cocoClassNames';
import { Detection, DetectionResult } from './types/objectDetection';
import { useImageProcessing } from './useImageProcessing';
import { useObjectAnalysis } from './useObjectAnalysis';

export function useObjectDetection(format: any) {
  const modelRef = useRef<TensorflowModel | null>(null);
  const cameraRef = useRef<Camera>(null);
  const frameCount = Worklets.createSharedValue(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [detectionResults, setDetectionResults] = useState<DetectionResult[]>(
    []
  );
  const [hasPermission, setHasPermission] = useState(false);
  const device = useCameraDevice('back');

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

      logWorklet('[Worklet] Model outputs debug:');
      logWorklet(`[Worklet] Boxes shape: ${boxes.length}`);
      logWorklet(`[Worklet] Classes shape: ${classes.length}`);
      logWorklet(`[Worklet] Scores shape: ${scores.length}`);
      logWorklet(
        `[Worklet] First few classes: ${Array.from(classes.slice(0, 5))}`
      );
      logWorklet(
        `[Worklet] First few scores: ${Array.from(scores.slice(0, 5))}`
      );

      const totalDetections = Math.min(
        Math.round(numDetections[0] || 0),
        scores.length
      );

      const detected: Detection[] = [];

      logWorklet(`[Worklet] Total detections: ${totalDetections}`);

      for (let i = 0; i < totalDetections; i++) {
        const score = scores[i];
        const classId = Math.round(classes[i]);
        const className = COCO_CLASS_NAMES[classId] || 'unknown';
        logWorklet(
          `[Worklet] Detection ${i} - Score: ${score.toFixed(
            4
          )}, Class: ${classId} (${className})`
        );

        if (score < CONFIDENCE_THRESHOLD) {
          logWorklet(
            `[Worklet] Detection ${i} skipped - Score below threshold`
          );
          continue;
        }

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

  const capturePhoto = async (detection: Detection): Promise<string | null> => {
    try {
      if (!cameraRef.current) {
        console.log('[JS] Camera ref is not available');
        return null;
      }

      console.log('[JS] Starting photo capture for detection:', {
        x: detection.x,
        y: detection.y,
        width: detection.width,
        height: detection.height,
        class_id: detection.class_id,
      });

      const photo = await cameraRef.current.takePhoto({
        flash: 'off',
        enableShutterSound: false,
      });

      console.log('[JS] Photo captured successfully:', {
        path: photo.path,
        width: photo.width,
        height: photo.height,
      });

      // 이미지 크기 조정
      const manipResult = await ImageManipulator.manipulateAsync(
        photo.path,
        [{ resize: { width: 800 } }],
        { compress: 0.7, format: ImageManipulator.SaveFormat.JPEG }
      );

      console.log('[JS] Image resized:', {
        originalPath: photo.path,
        newPath: manipResult.uri,
        width: manipResult.width,
        height: manipResult.height,
      });

      // 이미지 데이터를 base64로 변환
      const response = await fetch(manipResult.uri);
      const blob = await response.blob();

      console.log('[JS] Image blob created:', {
        size: blob.size,
        type: blob.type,
      });

      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const base64data = reader.result as string;
          const base64String = base64data.split(',')[1];
          console.log('[JS] Base64 conversion completed:', {
            length: base64String.length,
            preview: base64String.substring(0, 50) + '...',
          });
          resolve(base64String);
        };
        reader.onerror = (error) => {
          console.error('[JS] Base64 conversion error:', error);
          reject(error);
        };
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('[JS] Photo capture error:', error);
      return null;
    }
  };

  const processExtractedData = async (extractedDataArray: any[]) => {
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
      const promises = uniqueItems.map(async (item) => {
        const { detection, timestamp } = item;
        console.log('[JS] Processing item:', {
          detection,
          timestamp,
        });

        const base64String = await capturePhoto(detection);
        if (!base64String) {
          console.log('[JS] Failed to capture photo for detection');
          return null;
        }

        // Convert base64 string to Uint8Array
        const binaryString = atob(base64String);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        const result = await processImageData(
          bytes,
          detection,
          timestamp,
          true
        );

        if (!result) {
          console.log('[JS] Failed to process image data');
          return null;
        }

        console.log('[JS] Successfully processed detection:', {
          class_id: detection.class_id,
          sugar_content: result.detection.sugar_content,
        });

        return {
          ...result,
          imageData: `data:image/jpeg;base64,${base64String}`,
        };
      });

      const results = await Promise.allSettled(promises);
      console.log('[JS] All promises settled:', {
        total: results.length,
        fulfilled: results.filter((r) => r.status === 'fulfilled').length,
        rejected: results.filter((r) => r.status === 'rejected').length,
      });

      const validResults = results
        .filter(
          (result): result is PromiseFulfilledResult<DetectionResult> =>
            result.status === 'fulfilled' && result.value !== null
        )
        .map((result) => result.value);

      console.log('[JS] Valid results:', {
        count: validResults.length,
        results: validResults.map((r) => ({
          class_id: r.detection.class_id,
          sugar_content: r.detection.sugar_content,
        })),
      });

      if (validResults.length > 0) {
        setDetectionResults((prev) => {
          const prevKeys = new Set(prev.map((r) => getGridKey(r.detection)));
          const newUnique = validResults.filter(
            (r) => !prevKeys.has(getGridKey(r.detection))
          );
          return [...prev, ...newUnique];
        });
      }
    } catch (error) {
      console.error('[JS] Batch processing error:', error);
    }
  };

  const runOnJSThread = Worklets.createRunOnJS(processExtractedData);

  const SAMPLE_RATE = 10;

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
