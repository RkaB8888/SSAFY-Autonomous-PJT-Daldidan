// daldidan/components/AnalyzedResultOverlay.tsx
// useAnalysisApiHandler 훅에서 올바른 배열과 원본 해상도를 넘겨준다면 이 코드는 정상 작동합니다.
// (변환 로직, 렌더링 로직 포함)

import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  useCallback,
} from "react";
import {
  Animated,
  StyleSheet,
  Text,
  View,
  Easing,
  Pressable,
  Image,
} from "react-native";
import { AnalyzedObjectResult } from "../hooks/types/objectDetection"; // 이 타입에 segmentation 필드 추가 필요
import VisualBar from "./VisualBar";
import {
  Canvas,
  Rect,
  Group,
  Skia,
  Path,
  SkPath,
} from "@shopify/react-native-skia"; // Path 추가
import InfoTooltip from "./InfoTooltip";
// import question_apple from "../assets/images/question_apple.png"; // 사용되지 않으므로 주석 처리 또는 삭제 가능
import ShakeReminder from "./ShakeReminder";
import AppleToastStack from "./AppleToastStack";
import TopNAppleSelector from "./TopNAppleSelector";
import AppleJuiceAnimation from "./AppleJuiceAnimation";
import { useInfoTooltip } from "./InfoTooltipContext";
import TopAppleCrown from "./TopAppleCrown";
import LottieView from "lottie-react-native";

interface Props {
  results: AnalyzedObjectResult[];
  screenSize: { width: number; height: number };
  originalImageSize: { width: number; height: number };
  minSugar: number;
  onChangeMinSugar: (s: number) => void;
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

const transformBboxToScreen = (
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number },
  originalWidth: number,
  originalHeight: number,
  screenWidth: number,
  screenHeight: number
) => {
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
};

export default function AnalyzedResultOverlay({
  results,
  screenSize,
  originalImageSize,
   minSugar,        
  onChangeMinSugar,
}: Props) {
  const [selectedAppleId, setSelectedAppleId] = useState<
    string | number | null
  >(null);
  const [juiceAnimations, setJuiceAnimations] = useState<
    {
      id: string;
      color: string;
      position: { x: number; y: number };
      size: number;
    }[]
  >([]);

  // type FilterMode = "topN" | "slider";
  // const [filterMode, setFilterMode] = useState<FilterMode>("topN");

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

  // 상단 useState 추가
  const { hasShown, setHasShown } = useInfoTooltip();
  const [showTooltip, setShowTooltip] = useState(false);

  useEffect(() => {
    if (!hasShown) {
      setShowTooltip(true);
      setHasShown(true);
    }
  }, [hasShown]);

  const scaleAnim = useRef(new Animated.Value(1)).current;

  // const [topN, setTopN] = useState(3);
  // const [minSugar, setMinSugar] = useState(11);

  // minSugar가 변경될 때 filterMode를 'slider'로 변경
  const stableResults = useMemo(() => results, [results]); // (거의 영향 없을 수도 있으나 넣어둠)
  const [showTopOnly, setShowTopOnly] = useState(false);

  const top3Ids = useMemo(
    () =>
      [...results]
        .filter((r) => r.sugar_content != null)
        .sort((a, b) => b.sugar_content! - a.sugar_content!)
        .slice(0, 3)
        .map((r) => r.id),
    [results]
  );
 const handleMinSugarChange = useCallback((sugar: number) => {
  onChangeMinSugar(sugar); // 부모에게 전달
}, [onChangeMinSugar]);

  // topN이 변경될 때 filterMode를 'topN'으로 변경
  // const handleTopNChange = (n: number) => {
  //   setTopN(n);
  //   setFilterMode("topN");
  // };

  // topNIds를 useMemo로 메모이제이션
  // const topNIds = useMemo(
  //   () =>
  //     [...results]
  //       .filter(
  //         (r) => r.sugar_content !== undefined && r.sugar_content !== null
  //       )
  //       .sort((a, b) => b.sugar_content! - a.sugar_content!)
  //       .slice(0, topN)
  //       .map((r) => r.id),
  //   [results, topN]
  // );

  const highest = useMemo(
    () =>
      [...results]
        .filter(
          (r) => r.sugar_content !== undefined && r.sugar_content !== null
        )
        .sort((a, b) => b.sugar_content! - a.sugar_content!)[0],
    [results]
  );

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

  // useEffect(() => {
  //   if (topN > results.length) {
  //     setTopN(results.length > 0 ? results.length : 1); // 최소 1개는 있도록 수정
  //   }
  // }, [results.length, topN]);

  const handleApplePress = (result: AnalyzedObjectResult) => {
    if (result.bbox) {
      const screenBbox = transformBboxToScreen(
        result.bbox,
        originalImageSize.width,
        originalImageSize.height,
        screenSize.width,
        screenSize.height
      );
      const centerX = (screenBbox.x1 + screenBbox.x2) / 2;
      const centerY = (screenBbox.y1 + screenBbox.y2) / 2;

      // 사과 객체의 크기 계산 (너비와 높이 중 더 큰 값 사용)
      const appleWidth = screenBbox.x2 - screenBbox.x1;
      const appleHeight = screenBbox.y2 - screenBbox.y1;
      const appleSize = Math.max(appleWidth, appleHeight) * 3.5;

      const animationId = `${result.id}-${Date.now()}`;
      setJuiceAnimations((prev) => [
        ...prev,
        {
          id: animationId,
          color: "#ff6b6b",
          position: { x: centerX, y: centerY },
          size: appleSize,
        },
      ]);
    }
  };

  // 세그멘테이션 선 스타일
  const SEGMENTATION_STROKE_WIDTH = 4; // 선 두께 (조절 가능)
  const SEGMENTATION_STROKE_COLOR = "rgba(209, 14, 14, 0.85)"; // 선 색상 (라임 그린, 반투명, 조절 가능)
  // 직선 보간 레벨 (0: 사용 안함, 1 이상: 해당 개수만큼 중간점 추가)
  // 동서남북 각진 부분을 완화하기 위해 Catmull-Rom을 직접 사용하거나, 약간의 보간 후 사용
  const LINEAR_INTERPOLATION_LEVEL = 1; // 0 또는 1로 테스트해보세요.

  // 애니메이션 ref 추가
  const animationRef = useRef<LottieView>(null);
  const [animationPositions, setAnimationPositions] = useState<
    Array<{ x: number; y: number; size: number }>
  >([]);

  const shouldShow = useCallback(
    (r: AnalyzedObjectResult) =>
      showTopOnly
        ? top3Ids.includes(r.id)
        : r.sugar_content != null && r.sugar_content >= minSugar,
    [showTopOnly, top3Ids, minSugar]
  );

  // 애니메이션 위치 업데이트를 위한 useEffect
 useEffect(() => {
  const newPositions: Array<{ x: number; y: number; size: number }> = [];

  results.forEach((result) => {
    if (!shouldShow(result) || !result.segmentation?.points?.length) return;

    const originalPoints = result.segmentation.points;
    let pointsToProcess = originalPoints;

    if (LINEAR_INTERPOLATION_LEVEL > 0 && originalPoints.length >= 2) {
      pointsToProcess = interpolateOriginalPoints(
        originalPoints,
        LINEAR_INTERPOLATION_LEVEL
      );
    }

    const screenPoints = pointsToProcess.map((p: number[]) =>
      transformPointToScreen(
        p,
        originalImageSize.width,
        originalImageSize.height,
        screenSize.width,
        screenSize.height
      )
    );

    const centerX =
      screenPoints.reduce((sum, p) => sum + p.x, 0) / screenPoints.length;
    const centerY =
      screenPoints.reduce((sum, p) => sum + p.y, 0) / screenPoints.length;

    const xCoords = screenPoints.map((p) => p.x);
    const yCoords = screenPoints.map((p) => p.y);
    const width = Math.max(...xCoords) - Math.min(...xCoords);
    const height = Math.max(...yCoords) - Math.min(...yCoords);
    const size = Math.max(width, height) * 1.5;

    newPositions.push({ x: centerX, y: centerY, size });
  });

  setAnimationPositions(newPositions);
}, [results, shouldShow, screenSize, originalImageSize]);


  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="auto">
      <View style={{ position: "absolute", top: 80, left: 27, zIndex: 1000 }}>
        <Pressable onPress={() => setShowTopOnly((prev) => !prev)}>
          <Image
            source={
              showTopOnly
                ? require("../assets/images/all_apple.png")
                : require("../assets/images/top3.png")
            }
            style={{ width: 60, height: 60, resizeMode: "contain" }}
          />
          <Text style={{ marginTop: 4, fontSize: 12, fontWeight: "bold", color: "#fff" }}>
        {showTopOnly ? "전체 보기" : "TOP3 보기"}
      </Text>
        </Pressable>
      </View>
      <View
        style={{
          position: "absolute",
          top: 40,
          right: 10,
          height: 300,
          width: 60,
          zIndex: 1000,
        }}
        pointerEvents="auto"
      >
        <View style={{ flex: 1 }} pointerEvents="auto">
          <VisualBar
            results={results}
            minSugar={minSugar}
            onChangeMinSugar={handleMinSugarChange}
          />
        </View>
      </View>

      <Canvas
        style={[StyleSheet.absoluteFill, { zIndex: 1 }]}
        pointerEvents="none"
      >
        {/* 전체 어두운 레이어 */}
        <Group>
          <Rect
            x={0}
            y={0}
            width={screenSize.width}
            height={screenSize.height}
            color="rgba(0, 0, 0, 0.5)" // 반투명 검은색
          />
        </Group>

        {/* 세그멘테이션 영역 클리핑 */}
        {results.map((result, index) => {
          if (!shouldShow(result) || !result.segmentation?.points?.length) return null;

          if (
            result.segmentation?.points &&
            result.segmentation.points.length > 0
          ) {
            const originalPoints = result.segmentation.points;
            let pointsToProcess = originalPoints;

            if (LINEAR_INTERPOLATION_LEVEL > 0 && originalPoints.length >= 2) {
              pointsToProcess = interpolateOriginalPoints(
                originalPoints,
                LINEAR_INTERPOLATION_LEVEL
              );
            }

            const screenPoints = pointsToProcess.map((p: number[]) =>
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
                  <Path path={skPath} color="rgba(0,0,0,0)" blendMode="clear" />
                  <Path
                    path={skPath}
                    style="stroke"
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

      {/* Canvas 밖에서 애니메이션 렌더링 */}
      {animationPositions.map((pos, index) => (
        <View
          key={`animation-${index}`}
          style={{
            position: "absolute",
            left: pos.x - pos.size / 2,
            top: pos.y - pos.size / 2,
            width: pos.size,
            height: pos.size,
            zIndex: 1000,
            pointerEvents: "none",
          }}
        >
          <LottieView
            ref={animationRef}
            source={require("../assets/lottie/tap.json")}
            autoPlay
            loop
            style={{ width: "100%", height: "100%" }}
            renderMode="AUTOMATIC"
            speed={1}
          />
        </View>
      ))}

      {/* ✅ 왕관은 여기! */}
      {highest?.bbox && (
        <TopAppleCrown
          bbox={highest.bbox}
          originalSize={originalImageSize}
          screenSize={screenSize}
        />
      )}

      <View
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          zIndex: 2,
        }}
        pointerEvents="auto"
      >
        <AppleToastStack
          results={results}
          screenSize={screenSize}
          originalImageSize={originalImageSize}
          onApplePress={handleApplePress}
        />
      </View>

      {juiceAnimations.map((animation) => (
        <AppleJuiceAnimation
          key={animation.id}
          color={animation.color}
          position={animation.position}
          size={animation.size}
          onAnimationEnd={() => {
            setJuiceAnimations((prev) =>
              prev.filter((a) => a.id !== animation.id)
            );
          }}
        />
      ))}

      <Animated.View
        style={[styles.infoButton, { transform: [{ scale: scaleAnim }] }]}
        pointerEvents="auto"
      >
        <Pressable onPress={() => setShowTooltip((prev) => !prev)}>
          <Image
            source={
              showTooltip
                ? require("../assets/images/explamation_apple.png")
                : require("../assets/images/question_apple.png")
            }
            style={styles.infoIcon}
          />
        </Pressable>
      </Animated.View>

      {/* ✅ 왕관은 여기! */}
      {highest?.bbox && !showTooltip && (
        <TopAppleCrown
          bbox={highest.bbox}
          originalSize={originalImageSize}
          screenSize={screenSize}
        />
      )}

      <ShakeReminder />

      {/* InfoTooltip을 최상위로 이동 */}
      {showTooltip && (
        <View
          style={{
            position: "absolute",
            width: "100%",
            height: "100%",
            zIndex: 9999,
          }}
          pointerEvents="auto"
        >
          <InfoTooltip onDismiss={() => setShowTooltip(false)} />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  infoButton: {
    position: "absolute",
    bottom: 20,
    right: 5,
    zIndex: 1000,
    elevation: 10,
  },
  infoIcon: {
    width: 88,
    height: 88,
    bottom: 20,
    resizeMode: "contain",
  },
});
