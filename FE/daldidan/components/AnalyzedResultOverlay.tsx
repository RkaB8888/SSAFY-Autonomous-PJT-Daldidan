// daldidan/components/AnalyzedResultOverlay.tsx
// useAnalysisApiHandler 훅에서 올바른 배열과 원본 해상도를 넘겨준다면 이 코드는 정상 작동합니다.
// (변환 로직, 렌더링 로직 포함)

import React, { useState } from "react";
import { useEffect, useRef } from "react";
import { Animated, StyleSheet, Text, View, Easing } from "react-native";
import { AnalyzedObjectResult } from "../hooks/types/objectDetection";
import VisualBar from "./VisualBar";
import { Canvas, Rect, Group, Skia } from "@shopify/react-native-skia";
import { Pressable } from "react-native";
import InfoTooltip from "./InfoTooltip"; // 상단에 import 추가
import question_apple from "../assets/images/question_apple.png";
import { Image } from "react-native"; // ✅ 추가
import ShakeReminder from "./ShakeReminder";
import AppleToastStack from "./AppleToastStack";

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
  const [selectedAppleId, setSelectedAppleId] = useState<
    string | number | null
  >(null);

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

  const [showTooltip, setShowTooltip] = useState(false);
  const scaleAnim = useRef(new Animated.Value(1)).current;

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
  }, []);

  const transformBboxToScreen = (
    bbox: { xmin: number; ymin: number; xmax: number; ymax: number },
    originalWidth: number, // 예: 1440
    originalHeight: number, // 예: 1080
    screenWidth: number, // 예: 360
    screenHeight: number // 예: 712
  ) => {
    // 1. 시계방향 90도 회전 (이미지는 landscape이므로 회전 필요)
    const rotatedX1 = originalHeight - bbox.ymax;
    const rotatedY1 = bbox.xmin;
    const rotatedX2 = originalHeight - bbox.ymin;
    const rotatedY2 = bbox.xmax;

    const rotatedImageWidth = originalHeight; // 1080
    const rotatedImageHeight = originalWidth; // 1440

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

  const handleApplePress = (appleId: string | number) => {
    setSelectedAppleId(appleId);
  };
  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
      <VisualBar results={results} onApplePress={handleApplePress} />
      {/* 🔶 Skia 마스킹 캔버스 */}
      <Canvas style={StyleSheet.absoluteFill}>
        {/* 전체 어두운 레이어 */}
        <Group>
          <Rect
            x={0}
            y={0}
            width={screenSize.width}
            height={screenSize.height}
            color="rgba(0, 0, 0, 0.5)"
          />
        </Group>

        {/* 바운딩 박스들 위에 투명한 박스 그려서 클리핑 */}
        {results.map((result, index) => {
          const screenBbox = transformBboxToScreen(
            result.bbox,
            originalImageSize.width,
            originalImageSize.height,
            screenSize.width,
            screenSize.height
          );
          const screenWidth = screenBbox.x2 - screenBbox.x1;
          const screenHeight = screenBbox.y2 - screenBbox.y1;

          return (
            <Rect
              key={`mask-${index}`}
              x={screenBbox.x1}
              y={screenBbox.y1}
              width={screenWidth}
              height={screenHeight}
              color="rgba(0, 0, 0, 0)"
              blendMode="clear" // 핵심! 이걸로 해당 영역만 비워줌
            />
          );
        })}
      </Canvas>

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

        // bbox 시각화용 디버그 뷰
        return (
          <React.Fragment key={result.id ?? index}>
            <View
              style={{
                position: "absolute",
                left: screenBbox.x1,
                top: screenBbox.y1,
                width: screenWidth,
                height: screenHeight,
                borderWidth: 2,
                borderColor: "rgba(255,0,0,0.5)",
                backgroundColor: "rgba(255,0,0,0.08)",
                zIndex: 100,
                pointerEvents: "none",
              }}
            />
          </React.Fragment>
        );
      })}
      <AppleToastStack
        results={results}
        screenSize={screenSize}
        originalImageSize={originalImageSize}
      />
      {/* ℹ️ 버튼 */}
      <Animated.View
        style={[
          styles.infoButton, // ✅ 위치를 여기로 옮김!
          { transform: [{ scale: scaleAnim }] },
        ]}
      >
        <Pressable
          onPress={() => setShowTooltip((prev) => !prev)}
          style={styles.infoButton}
        >
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

      {/* 모달 */}
      {showTooltip && <InfoTooltip onDismiss={() => setShowTooltip(false)} />}

      <ShakeReminder />
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
  selectedText: {
    color: "#000",
    fontWeight: "bold",
  },
  infoButton: {
    position: "absolute",
    bottom: 20,
    right: 5,
    zIndex: 1000,
    elevation: 10,
  },
  infoIcon: {
    width: 58,
    height: 68,
    resizeMode: "contain",
  },
});
