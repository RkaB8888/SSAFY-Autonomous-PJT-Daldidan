// daldidan/components/AnalyzedResultOverlay.tsx
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection'; // API 분석 결과 타입
import { COCO_CLASS_NAMES } from '../constants/cocoClassNames'; // 클래스 이름 필요시
// Skia를 사용하여 박스를 그릴 수도 있습니다. 필요에 따라 추가하세요.
// import { Canvas, Group, Rect } from '@shopify/react-native-skia';

interface Props {
  // useAnalysisApiHandler 훅에서 받아온 분석 결과 리스트 (null 아님이 보장됨)
  results: AnalyzedObjectResult[];
  // 카메라 뷰의 현재 화면 크기
  screenSize: { width: number; height: number };
  // TODO: API 분석 시 사용된 원본 이미지의 크기 정보도 필요할 수 있습니다.
  // 백엔드에서 반환하는 bbox 좌표가 어떤 이미지 크기 기준인지 확인하세요.
  // originalImageSize?: { width: number; height: number };
}

// API 분석 결과를 화면에 그리는 컴포넌트 (카메라 정지 상태에서 표시)
export default function AnalyzedResultOverlay({ results, screenSize /*, originalImageSize*/ }: Props) {

    if (!results || results.length === 0) {
        return null; // 결과가 없거나 빈 배열이면 아무것도 표시 안 함
    }

    // TODO: 여기에서 API 결과의 bbox 좌표를 화면 좌표계로 변환하는 로직을 구현해야 합니다.
    // 백엔드에서 받은 bbox 좌표가 (x1, y1, x2, y2) 형태이고 원본 이미지 크기 기준이라고 가정합니다.
    // 이 좌표를 RN 화면 크기 (screenSize.width, screenSize.height)에 맞게 스케일링해야 합니다.
    // 원본 이미지 크기 정보가 필요하다면 prop으로 받거나 API 응답에 포함되어야 합니다.
    // 카메라 뷰의 종횡비와 이미지의 종횡비가 다를 경우 추가적인 조정이 필요할 수 있습니다.

    // 아래는 예시 변환 로직입니다. 실제 환경에 맞춰 수정하세요.
    // 여기서는 원본 이미지 크기가 캡쳐 시 ViewShot을 감싼 View의 크기 (screenSize)와 동일하다고 가정한 간단한 예시입니다.
    const transformBboxToScreen = (bbox: { x1: number; y1: number; x2: number; y2: number; }, originalWidth: number, originalHeight: number, screenWidth: number, screenHeight: number) => {
        // 간단한 비율 스케일링 예시 (원본 이미지와 화면의 종횡비가 같다고 가정)
        const scaleX = screenWidth / originalWidth;
        const scaleY = screenHeight / originalHeight;

        return {
            x1: bbox.x1 * scaleX,
            y1: bbox.y1 * scaleY,
            x2: bbox.x2 * scaleX,
            y2: bbox.y2 * scaleY,
        };
    };

    // TODO: 실제 originalImageSize (API 분석에 사용된 원본 이미지의 크기)를 정의하거나 prop으로 받아야 합니다.
    // 임시로 screenSize와 같다고 가정합니다.
    const originalImageSize = screenSize; // ★★★ 이 부분은 실제 로직에 맞게 수정하세요! ★★★


  return (
    // 이 오버레이는 카메라 뷰 전체를 덮도록 절대 위치로 설정합니다.
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none"> {/* pointerEvents="box-none"으로 하위 터치 이벤트 통과 */}
        {/* TODO: Skia Canvas 또는 RN View/Text를 사용하여 바운딩 박스 및 텍스트 그리기 */}

        {results.map((result, index) => {
             // API 결과의 bbox 좌표 (result.bbox)를 화면 좌표로 변환합니다.
             const screenBbox = transformBboxToScreen(
                 result.bbox,
                 originalImageSize.width,
                 originalImageSize.height,
                 screenSize.width,
                 screenSize.height
             );

             const screenWidth = screenBbox.x2 - screenBbox.x1;
             const screenHeight = screenBbox.y2 - screenBbox.y1;

             // 객체 라벨 및 당도 텍스트
             const labelText = result.label || (result.class_id !== undefined ? COCO_CLASS_NAMES[result.class_id] : 'Unknown');
             const sugarText = result.sugar_content !== undefined && result.sugar_content !== null
                               ? `당도: ${result.sugar_content}Bx`
                               : '';
             const displayTexts = [labelText, sugarText].filter(Boolean).join(' - '); // 라벨과 당도 합치기


             // 객체 크기에 따라 텍스트 크기 조절 (선택 사항)
            const fontSize = Math.max(
              10,
              Math.min(14, Math.min(screenWidth, screenHeight) * 0.1) // 화면 크기 적용된 박스 크기 사용
            );


            return (
                <React.Fragment key={`analyzed-obj-${index}`}>
                    {/* 바운딩 박스 그리기 (예: RN View 스타일) */}
                    <View
                        style={{
                            position: 'absolute',
                            left: screenBbox.x1,
                            top: screenBbox.y1,
                            width: screenWidth,
                            height: screenHeight,
                            borderWidth: 2,
                            borderColor: 'yellow', // 분석 결과 박스 색상 (실시간과 다르게)
                            // backgroundColor: 'rgba(255, 255, 0, 0.2)', // 반투명 배경
                            zIndex: 5, // 다른 UI 위에 표시
                        }}
                    />
                    {/* 텍스트 라벨 및 당도 표시 */}
                     {displayTexts ? (
                       <View
                           style={[
                               styles.textContainer,
                               {
                                   position: 'absolute',
                                   // 텍스트 위치: 박스 좌상단 기준, 화면 범위 벗어나지 않도록 조정
                                   left: Math.max(0, Math.min(screenBbox.x1, screenSize.width - 150)),
                                   top: Math.max(0, Math.min(screenBbox.y1 - 25, screenSize.height - 25)),
                                   width: 150, // 텍스트 컨테이너 너비 고정
                                   backgroundColor: 'rgba(0, 0, 0, 0.7)',
                                   // borderColor: 'yellow', // 라벨 테두리 색상
                                   // borderWidth: 1,
                               },
                           ]}
                       >
                           <Text style={[styles.text, { fontSize }]} numberOfLines={1}>
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
    textContainer: {
        padding: 4,
        borderRadius: 4,
        zIndex: 6, // 바운딩 박스 View보다 위에
    },
    text: {
        color: 'white',
        fontWeight: 'bold',
        textAlign: 'center',
    },
});