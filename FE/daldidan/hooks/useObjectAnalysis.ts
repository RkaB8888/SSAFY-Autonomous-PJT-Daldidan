import { API_ENDPOINTS } from '../constants/api';
import base64 from 'base64-js';
import { DetectionResult } from './types/objectDetection';

export const useObjectAnalysis = () => {
  const sendDetectionToServer = async (
    base64String: string,
    classId: number | undefined
  ): Promise<any> => {
    const formData = new FormData();
    formData.append('image', base64String);
    formData.append('id', classId?.toString() ?? '1');

    const response = await fetch(API_ENDPOINTS.OBJECT_ANALYSIS, {
      method: 'POST',
      body: formData,
    });

    const text = await response.text();
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}, body: ${text}`);
    }

    try {
      return JSON.parse(text);
    } catch (e) {
      throw new Error('Response is not valid JSON');
    }
  };

  const processImageData = async (
    uint8Array: Uint8Array,
    detection: any,
    timestamp: number
  ): Promise<DetectionResult | null> => {
    try {
      const base64String = base64.fromByteArray(uint8Array);
      console.log('base64String', base64String);
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
          detection,
          imageData: `data:image/png;base64,${base64String}`,
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

  return {
    processImageData,
  };
};
