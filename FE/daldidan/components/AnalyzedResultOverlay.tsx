// daldidan/components/AnalyzedResultOverlay.tsx
// useAnalysisApiHandler 훅에서 올바른 배열과 원본 해상도를 넘겨준다면 이 코드는 정상 작동합니다.
// (변환 로직, 렌더링 로직 포함)

import React, { useState } from 'react';
import { useEffect, useRef } from 'react';
import {
  Animated,
  StyleSheet,
  Text,
  View,
  Easing,
  Pressable,
  Image,
} from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection'; // 이 타입에 segmentation 필드 추가 필요
import VisualBar from './VisualBar';
import {
  Canvas,
  Rect,
  Group,
  Skia,
  Path,
  SkPath,
} from '@shopify/react-native-skia'; // Path 추가
import InfoTooltip from './InfoTooltip';
// import question_apple from "../assets/images/question_apple.png"; // 사용되지 않으므로 주석 처리 또는 삭제 가능
import ShakeReminder from './ShakeReminder';
import AppleToastStack from './AppleToastStack';
import TopNAppleSelector from './TopNAppleSelector';

interface Props {
  results: AnalyzedObjectResult[];
  screenSize: { width: number; height: number };
  originalImageSize: { width: number; height: number };
}

// 각 포인트를 화면 좌표로 변환하는 함수
const transformPointToScreen = (
  point: number[], // [originalX, originalY]
  originalImageWidth: number,
  originalImageHeight: number,
  screenWidth: number,
  screenHeight: number
) => {
  const [originalX, originalY] = point;

  // 1. 시계방향 90도 회전 (이미지는 landscape이므로 회전 필요)
  // 원본 이미지의 (0,0)이 왼쪽 상단이라고 가정
  const rotatedX = originalImageHeight - originalY;
  const rotatedY = originalX;

  const rotatedImageActualWidth = originalImageHeight; // 회전 후 이미지의 너비는 원본 이미지의 높이
  const rotatedImageActualHeight = originalImageWidth; // 회전 후 이미지의 높이는 원본 이미지의 너비

  // 2. 화면 비율에 맞는 단일 scale 계산 (비율 유지, 화면 높이에 맞춤)
  // contentMode 'aspectFit' 또는 'contain' 과 유사한 효과
  const scale = screenHeight / rotatedImageActualHeight;

  // 3. 중심 정렬을 위한 패딩(offset) 계산
  // 스케일링된 이미지의 너비
  const scaledImageWidth = rotatedImageActualWidth * scale;
  const offsetX = (screenWidth - scaledImageWidth) / 2;

  // 스케일링된 이미지의 높이 (screenHeight와 동일해야 함)
  // const scaledImageHeight = rotatedImageActualHeight * scale;
  // const offsetY = (screenHeight - scaledImageHeight) / 2; // offsetY는 0이 될 것 (screenHeight에 맞췄으므로)
  const offsetY = 0;

  // 4. 최종 화면 좌표 변환
  return {
    x: Math.floor(rotatedX * scale + offsetX),
    y: Math.floor(rotatedY * scale + offsetY), // 위에서 계산된 offsetY 적용 (0일 가능성 높음)
  };
};

/**
 * 원본 폴리곤 점들 사이에 추가적인 점들을 보간하여 삽입합니다.
 * @param originalPoints 원본 폴리곤 점들의 배열 (예: [[x1, y1], [x2, y2], ...])
 * @param interpolationLevel 각 원본 점 쌍 사이에 추가할 점의 수. 0이면 원본 그대로, 1이면 각 쌍 사이에 1개의 중간점 추가.
 * @returns 보간된 점들을 포함한 새로운 폴리곤 점들의 배열
 */
const interpolateOriginalPoints = (
  originalPoints: number[][],
  interpolationLevel: number = 1
): number[][] => {
  // interpolationLevel이 0보다 작거나, 점이 2개 미만이면 보간 의미 없음
  if (interpolationLevel < 1 || originalPoints.length < 2) {
    return originalPoints;
  }

  const interpolatedPathPoints: number[][] = [];
  for (let i = 0; i < originalPoints.length; i++) {
    const p1 = originalPoints[i];
    // 폴리곤을 닫기 위해 마지막 점에서 첫 번째 점으로 연결
    const p2 = originalPoints[(i + 1) % originalPoints.length];

    interpolatedPathPoints.push(p1); // 현재 원본 점 추가

    // p1과 p2 사이에 'interpolationLevel' 개의 점 추가
    for (let j = 1; j <= interpolationLevel; j++) {
      const t = j / (interpolationLevel + 1); // 보간 계수 (0과 1 사이)
      const interpolatedX = p1[0] * (1 - t) + p2[0] * t;
      const interpolatedY = p1[1] * (1 - t) + p2[1] * t;
      interpolatedPathPoints.push([interpolatedX, interpolatedY]);
    }
  }
  return interpolatedPathPoints;
};

/**
 * 화면 좌표 점들을 기반으로 Catmull-Rom 스플라인을 이용한 부드러운 Skia Path(닫힌 폴리곤)를 생성합니다.
 * @param screenPoints 화면 좌표 점들의 배열 ( {x, y} 객체 형태 )
 * @returns SkPath 객체
 */
const createSkiaPathForClosedPolygonCurves = (
  screenPoints: { x: number; y: number }[]
): SkPath => {
  // 타입 변경
  const path = Skia.Path.Make(); // Skia.Path.Make()는 SkPath를 반환합니다.
  const n = screenPoints.length;

  if (n === 0) return path;
  if (n < 3) {
    path.moveTo(screenPoints[0].x, screenPoints[0].y);
    for (let i = 1; i < n; i++) {
      path.lineTo(screenPoints[i].x, screenPoints[i].y);
    }
    if (n > 0) path.close();
    return path;
  }

  path.moveTo(screenPoints[0].x, screenPoints[0].y);

  for (let i = 0; i < n; i++) {
    const p0 = screenPoints[(i - 1 + n) % n];
    const p1 = screenPoints[i];
    const p2 = screenPoints[(i + 1) % n];
    const p3 = screenPoints[(i + 2) % n];

    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;

    path.cubicTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
  path.close();
  return path;
};

export default function AnalyzedResultOverlay({
  results,
  screenSize,
  originalImageSize,
}: Props) {
  const [selectedAppleId, setSelectedAppleId] = useState<
    string | number | null
  >(null);

  type FilterMode = 'topN' | 'slider';
  const [filterMode, setFilterMode] = useState<FilterMode>('topN');

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
      '[AnalyzedResultOverlay] Not rendering: results empty or size info missing.',
      { results, screenSize, originalImageSize }
    );
    return null;
  }

  const [showTooltip, setShowTooltip] = useState(false);
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const [topN, setTopN] = useState(3);
  const [minSugar, setMinSugar] = useState(10);

  const topNIds = [...results]
    .filter((r) => r.sugar_content !== undefined && r.sugar_content !== null)
    .sort((a, b) => b.sugar_content! - a.sugar_content!)
    .slice(0, topN)
    .map((r) => r.id);

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(scaleAnim, {
          toValue: 1.1,
          duration: 500,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(scaleAnim, {
          toValue: 1,
          duration: 500,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, [scaleAnim]);

  useEffect(() => {
    if (topN > results.length) {
      setTopN(results.length > 0 ? results.length : 1); // 최소 1개는 있도록 수정
    }
  }, [results.length, topN]);

  // 기존 transformBboxToScreen 함수는 이제 사용되지 않으므로 삭제하거나 주석 처리 가능
  // const transformBboxToScreen = ( ... ) => { ... };

  // handleApplePress 함수는 현재 사용되지 않지만, 추후 개별 사과 선택 기능에 필요할 수 있어 유지
  // const handleApplePress = (appleId: string | number) => {
  //   setSelectedAppleId(appleId);
  // };

  // 세그멘테이션 선 스타일
  const SEGMENTATION_STROKE_WIDTH = 4; // 선 두께 (조절 가능)
  const SEGMENTATION_STROKE_COLOR = 'rgba(209, 14, 14, 0.85)'; // 선 색상 (라임 그린, 반투명, 조절 가능)
  // 직선 보간 레벨 (0: 사용 안함, 1 이상: 해당 개수만큼 중간점 추가)
  // 동서남북 각진 부분을 완화하기 위해 Catmull-Rom을 직접 사용하거나, 약간의 보간 후 사용
  const LINEAR_INTERPOLATION_LEVEL = 1; // 0 또는 1로 테스트해보세요.

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents='box-none'>
      <View
        style={{
          flexDirection: 'row',
          justifyContent: 'center',
          marginTop: 50,
          zIndex: 10,
        }}
      >
        <Pressable
          onPress={() => setFilterMode('topN')}
          style={{
            paddingHorizontal: 12,
            paddingVertical: 6,
            marginRight: 10,
            backgroundColor: filterMode === 'topN' ? '#ff8c00' : '#e0e0e0',
            borderRadius: 8,
          }}
        >
          <Text
            style={{
              fontWeight: 'bold',
              color: filterMode === 'topN' ? 'white' : 'black',
            }}
          >
            TopN 모드
          </Text>
        </Pressable>
        <Pressable
          onPress={() => setFilterMode('slider')}
          style={{
            paddingHorizontal: 12,
            paddingVertical: 6,
            backgroundColor: filterMode === 'slider' ? '#ff8c00' : '#e0e0e0',
            borderRadius: 8,
          }}
        >
          <Text
            style={{
              fontWeight: 'bold',
              color: filterMode === 'slider' ? 'white' : 'black',
            }}
          >
            최소 당도 모드
          </Text>
        </Pressable>
      </View>

      {filterMode === 'topN' && (
        <TopNAppleSelector
          topN={topN}
          onChange={setTopN}
          maxN={Math.max(1, results.length)}
        />
      )}

      {filterMode === 'slider' && (
        <VisualBar
          results={results}
          minSugar={minSugar}
          onChangeMinSugar={setMinSugar}
        />
      )}

      <Canvas style={StyleSheet.absoluteFill}>
        {/* 전체 어두운 레이어 */}
        <Group>
          <Rect
            x={0}
            y={0}
            width={screenSize.width}
            height={screenSize.height}
            color='rgba(0, 0, 0, 0.5)' // 반투명 검은색
          />
        </Group>

        {/* 세그멘테이션 영역 클리핑 */}
        {results.map((result, index) => {
          const isHighlighted =
            filterMode === 'topN'
              ? topNIds.includes(result.id)
              : result.sugar_content !== undefined &&
                result.sugar_content !== null &&
                result.sugar_content >= minSugar;

          if (
            isHighlighted &&
            result.segmentation &&
            result.segmentation.points &&
            result.segmentation.points.length > 0 // 최소 1개의 점이라도 있어야 함 (Path 함수 내부에서 3개 미만 처리)
          ) {
            const originalPoints = result.segmentation.points;
            let pointsToProcess = originalPoints;

            if (LINEAR_INTERPOLATION_LEVEL > 0 && originalPoints.length >= 2) {
              // 2개 이상 점이 있을 때 보간 가능
              pointsToProcess = interpolateOriginalPoints(
                originalPoints,
                LINEAR_INTERPOLATION_LEVEL
              );
            }

            const screenPoints = pointsToProcess.map((p) =>
              transformPointToScreen(
                p,
                originalImageSize.width,
                originalImageSize.height,
                screenSize.width,
                screenSize.height
              )
            );

            const skPath: SkPath =
              createSkiaPathForClosedPolygonCurves(screenPoints);
            if (skPath.countPoints() > 0) {
              return (
                <React.Fragment key={`segment-group-${result.id}-${index}`}>
                  <Path path={skPath} color='rgba(0,0,0,0)' blendMode='clear' />
                  <Path
                    path={skPath}
                    style='stroke'
                    strokeWidth={SEGMENTATION_STROKE_WIDTH}
                    color={SEGMENTATION_STROKE_COLOR}
                  />
                </React.Fragment>
              );
            }
          }
          return null;
        })}
      </Canvas>

      {/* 바운딩 박스 시각화용 디버그 View 삭제됨 */}

      <AppleToastStack
        results={results}
        screenSize={screenSize}
        originalImageSize={originalImageSize}
        // transformPointFunction={transformPointToScreen} // 필요하다면 토스트 위치 계산에 사용
      />

      <Animated.View
        style={[styles.infoButton, { transform: [{ scale: scaleAnim }] }]}
      >
        <Pressable
          onPress={() => setShowTooltip((prev) => !prev)}
          // style={styles.infoButton} // Animated.View에 스타일이 이미 적용됨
        >
          <Image
            source={
              showTooltip
                ? require('../assets/images/explamation_apple.png')
                : require('../assets/images/question_apple.png')
            }
            style={styles.infoIcon}
          />
        </Pressable>
      </Animated.View>

      {showTooltip && <InfoTooltip onDismiss={() => setShowTooltip(false)} />}
      <ShakeReminder />
    </View>
  );
}

const styles = StyleSheet.create({
  // textContainer, text, selectedText는 현재 코드에서 직접 사용되지 않음
  // textContainer: {},
  // text: {
  //   color: "white",
  //   fontWeight: "bold",
  //   textAlign: "center",
  // },
  // selectedText: {
  //   color: "#000",
  //   fontWeight: "bold",
  // },
  infoButton: {
    position: 'absolute',
    bottom: 20,
    right: 5,
    zIndex: 1000, // AppleToastStack 보다 위에 오도록 충분히 높은 값
    elevation: 10,
  },
  infoIcon: {
    width: 58,
    height: 68,
    resizeMode: 'contain',
  },
});
