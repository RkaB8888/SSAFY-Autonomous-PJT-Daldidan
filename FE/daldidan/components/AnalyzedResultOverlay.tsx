// daldidan/components/AnalyzedResultOverlay.tsx
// useAnalysisApiHandler 훅에서 올바른 배열과 원본 해상도를 넘겨준다면 이 코드는 정상 작동합니다.
// (변환 로직, 렌더링 로직 포함)

import React from "react";
import { StyleSheet, Text, View, Dimensions } from "react-native";
import { AnalyzedObjectResult } from "../hooks/types/objectDetection";
import { COCO_CLASS_NAMES } from "../constants/cocoClassNames";

interface Props {
  // useAnalysisApiHandler 훅에서 받아온 분석 결과 리스트 (null 아님이 상위에서 보장됨)
  // 훅에서 올바른 배열(AnalyzedObjectResult[])을 전달할 것이라고 가정합니다.
  results: AnalyzedObjectResult[];
  // 카메라 뷰의 현재 화면 크기
  screenSize: { width: number; height: number };
  // API 분석 시 사용된 원본 이미지의 크기 정보 (필수)
  // 훅에서 올바른 객체({ width, height })를 전달할 것이라고 가정합니다.
  originalImageSize: { width: number; height: number };
}

export default function AnalyzedResultOverlay({
  results,
  screenSize,
  originalImageSize,
}: Props) {
  // results가 null이거나 비어있으면 렌더링 안 함 (훅에서 제대로 넘겨준다면 이 체크는 통과될 것입니다)
  if (
    !results ||
    results.length === 0 ||
    !screenSize ||
    screenSize.width <= 0 ||
    screenSize.height <= 0 ||
    !originalImageSize ||
    originalImageSize.width <= 0 ||
    originalImageSize.height <= 0
  ) {
    console.log(
      "[AnalyzedResultOverlay] Not rendering: results empty or size info missing.",
      { results, screenSize, originalImageSize }
    );
    return null;
  }

const transformBboxToScreen = (
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number },
  originalWidth: number,   // 예: 1440
  originalHeight: number,  // 예: 1080
  screenWidth: number,     // 예: 360
  screenHeight: number     // 예: 712
) => {
  // 1. 시계방향 90도 회전 (이미지는 landscape이므로 회전 필요)
    const rotatedX1 = originalHeight - bbox.ymax;
    const rotatedY1 = bbox.xmin;
    const rotatedX2 = originalHeight - bbox.ymin;
    const rotatedY2 = bbox.xmax;

  const rotatedImageWidth = originalHeight;  // 1080
  const rotatedImageHeight = originalWidth;  // 1440

  // 2. 화면 비율에 맞는 단일 scale 계산 (비율 유지)
const scale = screenHeight / rotatedImageHeight;

  // 3. 중심 정렬을 위한 패딩 계산
  const offsetX = (screenWidth - rotatedImageWidth * scale) / 2;
  const offsetY = (screenHeight - rotatedImageHeight * scale) / 2;

  // 4. 최종 화면 좌표 변환
    return {
    x1: Math.floor(rotatedX1 * scale + offsetX),
    y1: Math.floor(rotatedY1 * scale + offsetY),
    x2: Math.ceil(rotatedX2 * scale + offsetX),
    y2: Math.ceil(rotatedY2 * scale + offsetY),
    };
};


  return (
  <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
    {results.map((result, index) => {
      const screenBbox = transformBboxToScreen(
        result.bbox,
        originalImageSize.width,
        originalImageSize.height,
        screenSize.width,
        screenSize.height
      );
      

      const screenWidth = Math.max(0, screenBbox.x2 - screenBbox.x1);
      const screenHeight = Math.max(0, screenBbox.y2 - screenBbox.y1);

      console.log('[📦 bbox]', result.bbox);
        console.log('[📐 screenSize]', screenSize);
        console.log('[📷 originalSize]', originalImageSize);
        console.log('[📦 screenBbox]', screenBbox);
        console.log('[📏 boxWidth, boxHeight]', screenWidth, screenHeight);

      const labelText =
        result.id !== undefined ? `객체 ${result.id}` : `객체 ${index + 1}`;
      const sugarText =
        result.sugar_content !== undefined && result.sugar_content !== null
          ? `당도: ${result.sugar_content.toFixed(1)}Bx`
          : "";
      const displayTexts = [labelText, sugarText].filter(Boolean).join(" - ");

      const fontSize = Math.max(
        10,
        Math.min(14, Math.min(screenWidth, screenHeight) * 0.1)
      );

      return (
        <React.Fragment key={`analyzed-obj-${result.id ?? index}`}>
          {/* 바운딩 박스 */}
          {screenWidth > 0 &&
          screenHeight > 0 &&
          screenBbox.x1 >= 0 &&
          screenBbox.y1 >= 0 &&
          screenBbox.x2 <= screenSize.width &&
          screenBbox.y2 <= screenSize.height ? (
            <View
              style={{
                position: "absolute",
                left: screenBbox.x1,
                top: screenBbox.y1,
                width: screenWidth,
                height: screenHeight,
                borderWidth: 2,
                borderColor: "yellow",
                zIndex: 5,
              }}
            />
          ) : null}

          {/* 텍스트 라벨 */}
          {displayTexts ? (
            <View
              style={[
                styles.textContainer,
                {
                  position: "absolute",
                  left: Math.max(
                    0,
                    Math.min(screenBbox.x1, screenSize.width - 150)
                  ),
                  top:
                    screenBbox.y1 - 30 > 0
                      ? screenBbox.y1 - 30
                      : screenBbox.y2 + 5,
                  width: 150,
                  backgroundColor: "rgba(0, 0, 0, 0.7)",
                  padding: 4,
                  borderRadius: 4,
                  zIndex: 6,
                },
              ]}
            >
              <Text style={[{ fontSize }, styles.text]} numberOfLines={1}>
                {displayTexts}
              </Text>
            </View>
          ) : null}
        </React.Fragment>
      );
    })}
  </View>
);
}

const styles = StyleSheet.create({
  textContainer: {},
  text: {
    color: "white",
    fontWeight: "bold",
    textAlign: "center",
  },
});
