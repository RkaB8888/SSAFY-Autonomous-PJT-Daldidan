import React, { useState } from 'react';
import { Pressable } from 'react-native';
import AppleSugarToast from './AppleSugarToast';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';
import * as Haptics from 'expo-haptics';

interface Props {
  results: AnalyzedObjectResult[];
  screenSize: { width: number; height: number };
  originalImageSize: { width: number; height: number };
  onApplePress?: (result: AnalyzedObjectResult) => void;
}

function transformBboxToScreen(
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number },
  originalWidth: number,
  originalHeight: number,
  screenWidth: number,
  screenHeight: number
) {
  const rotatedX1 = originalHeight - bbox.ymax;
  const rotatedY1 = bbox.xmin;
  const rotatedX2 = originalHeight - bbox.ymin;
  const rotatedY2 = bbox.xmax;
  const rotatedImageWidth = originalHeight;
  const rotatedImageHeight = originalWidth;
  const scale = screenHeight / rotatedImageHeight;
  const offsetX = (screenWidth - rotatedImageWidth * scale) / 2;
  const offsetY = (screenHeight - rotatedImageHeight * scale) / 2;
  return {
    x1: Math.floor(rotatedX1 * scale + offsetX),
    y1: Math.floor(rotatedY1 * scale + offsetY),
    x2: Math.ceil(rotatedX2 * scale + offsetX),
    y2: Math.ceil(rotatedY2 * scale + offsetY),
  };
}

function generateId() {
  return Date.now().toString() + Math.random().toString(36).slice(2);
}

export default function AppleToastStack({
  results,
  screenSize,
  originalImageSize,
  onApplePress,
}: Props) {
  const [toasts, setToasts] = useState<
    {
      id: string;
      sugarContent: string;
      position: { x: number; y: number };
    }[]
  >([]);
  const [toastWidths, setToastWidths] = useState<{ [id: string]: number }>({});

  return (
    <>
      {results.map((result, index) => {
        const screenBbox = transformBboxToScreen(
          result.bbox ?? { xmin: 0, ymin: 0, xmax: 0, ymax: 0 },
          originalImageSize.width,
          originalImageSize.height,
          screenSize.width,
          screenSize.height
        );
        const screenWidth = Math.max(0, screenBbox.x2 - screenBbox.x1);
        const screenHeight = Math.max(0, screenBbox.y2 - screenBbox.y1);
        const baseX = screenBbox.x1 + screenWidth / 2;
        const baseY = Math.max(0, screenBbox.y1 - 40);
        const toastCount = toasts.filter(
          (t) => Math.abs(t.position.x - baseX) < 2
        ).length;
        return (
          <Pressable
            key={result.id ?? index}
            style={{
              position: 'absolute',
              left: screenBbox.x1,
              top: screenBbox.y1,
              width: screenWidth,
              height: screenHeight,
              zIndex: 5,
            }}
            onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
              onApplePress?.(result);
              setToasts((prev) => {
                const id = generateId();
                return [
                  ...prev,
                  {
                    id,
                    sugarContent: result.sugar_content?.toFixed(1) ?? '-',
                    position: {
                      x: baseX,
                      y: baseY - toastCount * 38,
                    },
                  },
                ];
              });
            }}
          />
        );
      })}
      {toasts.map((toast) => {
        const width = toastWidths[toast.id] ?? 80;
        return (
          <AppleSugarToast
            key={toast.id}
            visible={true}
            sugarContent={toast.sugarContent}
            position={{
              x: toast.position.x - width / 2,
              y: toast.position.y,
            }}
            onHide={() =>
              setToasts((prev) => prev.filter((t) => t.id !== toast.id))
            }
            onLayoutMeasured={(w: number) =>
              setToastWidths((prev) => ({ ...prev, [toast.id]: w }))
            }
          />
        );
      })}
    </>
  );
}
