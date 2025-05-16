// daldidan/components/AnalyzedResultOverlay.tsx
// useAnalysisApiHandler 훅에서 올바른 배열과 원본 해상도를 넘겨준다면 이 코드는 정상 작동합니다.
// (변환 로직, 렌더링 로직 포함)

import React from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';
import { COCO_CLASS_NAMES } from '../constants/cocoClassNames';

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

export default function AnalyzedResultOverlay({ results, screenSize, originalImageSize }: Props) {

    // results가 null이거나 비어있으면 렌더링 안 함 (훅에서 제대로 넘겨준다면 이 체크는 통과될 것입니다)
    if (!results || results.length === 0 || !screenSize || screenSize.width <= 0 || screenSize.height <= 0 || !originalImageSize || originalImageSize.width <= 0 || originalImageSize.height <= 0) {
        console.log("[AnalyzedResultOverlay] Not rendering: results empty or size info missing.", { results, screenSize, originalImageSize });
        return null;
    }

    // API 결과의 bbox 좌표 (xmin, ymin, xmax, ymax)를 현재 화면 좌표계로 변환하는 함수
    const transformBboxToScreen = (
        bbox: { xmin: number; ymin: number; xmax: number; ymax: number; }, // 백엔드 bbox 형태
        originalWidth: number, // 원본 이미지 너비 (훅에서 받아온 prop)
        originalHeight: number, // 원본 이미지 높이 (훅에서 받아온 prop)
        screenWidth: number, // 현재 화면 너비
        screenHeight: number // 현재 화면 높이
    ) => {
        // 1. EfficientDet은 중앙 크롭 기반으로 320x320으로 리사이즈 됨
        const cropSize = Math.min(originalWidth, originalHeight); // ex: 1440
        const cropOffsetX = (originalWidth - cropSize) / 2; // 0
        const cropOffsetY = (originalHeight - cropSize) / 2; // ex: 560

        // 2. bbox는 크롭된 기준에서 좌표로 해석됨 (그래야 model과 동일)
        const cropX1 = bbox.xmin - cropOffsetX;
        const cropY1 = bbox.ymin - cropOffsetY;
        const cropX2 = bbox.xmax - cropOffsetX;
        const cropY2 = bbox.ymax - cropOffsetY;

        // 3. 90도 회전 보정 (시계방향)
        const rotatedX1 = cropY1;
        const rotatedY1 = cropSize - cropX2;
        const rotatedX2 = cropY2;
        const rotatedY2 = cropSize - cropX1;

        // 4. 화면 비율로 스케일
        const scaleX = screenWidth / cropSize;
        const scaleY = screenHeight / cropSize;

        const screenX1 = rotatedX1 * scaleX;
        const screenY1 = rotatedY1 * scaleY;
        const screenX2 = rotatedX2 * scaleX;
        const screenY2 = rotatedY2 * scaleY;

        return {
            x1: Math.round(screenX1),
            y1: Math.round(screenY1),
            x2: Math.round(screenX2),
            y2: Math.round(screenY2),
        };
        };

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
        {/* results 배열을 순회하며 각 객체의 바운딩 박스와 텍스트를 렌더링 */}
        {results.map((result, index) => {
             // 여기서 result는 AnalyzedObjectResult 타입이며, result.bbox는 { xmin, ymin, xmax, ymax } 형태입니다.
             const screenBbox = transformBboxToScreen(
                 result.bbox, // { xmin, ymin, xmax, ymax } 형태
                 originalImageSize.width, // 원본 이미지 너비 (훅에서 받아온 prop)
                 originalImageSize.height, // 원본 이미지 높이 (훅에서 받아온 prop)
                 screenSize.width, // 화면 너비 (prop)
                 screenSize.height // 화면 높이 (prop)
             );

             const screenWidth = Math.max(0, screenBbox.x2 - screenBbox.x1);
             const screenHeight = Math.max(0, screenBbox.y2 - screenBbox.y1);

             const labelText = result.id !== undefined ? `객체 ${result.id}` : `객체 ${index + 1}`;
             const sugarText = result.sugar_content !== undefined && result.sugar_content !== null
                               ? `당도: ${result.sugar_content.toFixed(1)}Bx`
                               : '';
             const displayTexts = [labelText, sugarText].filter(Boolean).join(' - ');

             const fontSize = Math.max(
               10,
               Math.min(14, Math.min(screenWidth > 0 ? screenWidth : 1, screenHeight > 0 ? screenHeight : 1) * 0.1)
             );

            return (
                <React.Fragment key={`analyzed-obj-${result.id ?? index}`}>
                    {/* 바운딩 박스 그리기 */}
                    {/* 박스 크기나 위치가 유효하면 렌더링 */}
                    {screenWidth > 0 && screenHeight > 0 && screenBbox.x1 >= 0 && screenBbox.y1 >= 0 && screenBbox.x2 <= screenSize.width && screenBbox.y2 <= screenSize.height ? (
                        <View
                            style={{
                                position: 'absolute',
                                left: screenBbox.x1,
                                top: screenBbox.y1,
                                width: screenWidth,
                                height: screenHeight,
                                borderWidth: 2,
                                borderColor: 'yellow',
                                zIndex: 5,
                            }}
                        />
                    ) : null}
                    {/* 텍스트 라벨 및 당도 표시 */}
                     {displayTexts ? (
                       <View
                           style={[
                               styles.textContainer,
                               {
                                   position: 'absolute',
                                   left: Math.max(0, Math.min(screenBbox.x1, screenSize.width - 150)),
                                   top: Math.max(0, Math.min(screenBbox.y1 - 25, screenSize.height - 25)),
                                   width: 150,
                                   backgroundColor: 'rgba(0, 0, 0, 0.7)',
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
    textContainer: { },
    text: {
        color: 'white',
        fontWeight: 'bold',
        textAlign: 'center',
    },
});