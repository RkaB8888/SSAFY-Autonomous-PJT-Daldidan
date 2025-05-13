import { API_ENDPOINTS } from '../constants/api';
import base64 from 'base64-js';
import { DetectionResult } from './types/objectDetection';

export const useObjectAnalysis = () => {
  const sendDetectionToServer = async (
    base64String: string,
    classId: number | undefined
  ): Promise<any> => {
    const formData = new FormData();
    formData.append('image_base64', base64String);
    console.log(base64String);
    formData.append('id', classId?.toString() ?? '1');

    const response = await fetch(API_ENDPOINTS.OBJECT_ANALYSIS, {
      method: 'POST',
      body: formData,
    });

    const text = await response.text();
    console.log('API Response:', text);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
    }

    try {
      const parsedResult = JSON.parse(text);
      console.log('Parsed API Result:', parsedResult);
      return parsedResult;
    } catch (e) {
      console.error('JSON Parse Error:', e);
      throw new Error('Response is not valid JSON');
    }
  };

  const processImageData = async (
    uint8Array: Uint8Array,
    detection: any,
    timestamp: number,
    isJPEG: boolean = false
  ): Promise<DetectionResult | null> => {
    try {
      const base64String = base64.fromByteArray(uint8Array);
      if (!base64String || base64String.length === 0) {
        console.warn('[JS] Invalid base64 string generated');
        return null;
      }

      const result = await sendDetectionToServer(
        base64String,
        detection.class_id
      );

      if (result) {
        return {
          detection: {
            ...detection,
            sugar_content: result.sugar_content,
          },
          imageData: `data:image/${
            isJPEG ? 'jpeg' : 'png'
          };base64,${base64String}`,
          result,
          timestamp,
        };
      }
      return null;
    } catch (error) {
      console.error('[JS] API request error:', error);
      return null;
    }
  };

  const processBatch = async (
    items: Array<{
      detection: any;
      croppedData: { data: number[]; isJPEG?: boolean } | null;
      timestamp: number;
    }>
  ): Promise<DetectionResult[]> => {
    const promises = items.map(async (item) => {
      if (
        !item.croppedData ||
        !item.croppedData.data ||
        !Array.isArray(item.croppedData.data)
      ) {
        return null;
      }

      const uint8Array = new Uint8Array(item.croppedData.data);
      if (uint8Array.length === 0) return null;

      try {
        return await processImageData(
          uint8Array,
          item.detection,
          item.timestamp,
          item.croppedData.isJPEG
        );
      } catch (error) {
        console.error('[JS] Processing error:', error);
        return null;
      }
    });

    const results = await Promise.allSettled(promises);
    const validResults = results
      .filter(
        (result): result is PromiseFulfilledResult<DetectionResult> =>
          result.status === 'fulfilled' && result.value !== null
      )
      .map((result) => result.value);

    return validResults;
  };

  return {
    processImageData,
    processBatch,
  };
};
