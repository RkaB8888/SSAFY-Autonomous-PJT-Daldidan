// daldidan/components/AnalyzedResultOverlay.tsx
// useAnalysisApiHandler 훅에서 올바른 배열과 원본 해상도를 넘겨준다면 이 코드는 정상 작동합니다.
// (변환 로직, 렌더링 로직 포함)

import React, { useState } from 'react';
import { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View, Easing } from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';
import VisualBar from './VisualBar';
import { Canvas, Rect, Group, Skia } from '@shopify/react-native-skia';
import { Pressable } from 'react-native';
import InfoTooltip from './InfoTooltip'; // 상단에 import 추가
import question_apple from '../assets/images/question_apple.png';
import { Image } from 'react-native'; // ✅ 추가
import ShakeReminder from './ShakeReminder';
import AppleToastStack from './AppleToastStack';
import TopNAppleSelector from './TopNAppleSelector'; // topN 사과 선택 드롭다운 코드
import * as Haptics from 'expo-haptics';

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

  type FilterMode = 'topN' | 'slider';

  const [filterMode, setFilterMode] = useState<FilterMode>('topN');

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
      '[AnalyzedResultOverlay] Not rendering: results empty or size info missing.',
      { results, screenSize, originalImageSize }
    );
    return null;
  }

  const [showTooltip, setShowTooltip] = useState(false);
  const scaleAnim = useRef(new Animated.Value(1)).current;

  const [topN, setTopN] = useState(3); // 기본 top N : 3개

  const [minSugar, setMinSugar] = useState(10); // 슬라이더로 설정할 최소 당도 값(기본 최소값 10Bx)

  const topNIds = [...results]
    .filter((r) => r.sugar_content !== undefined && r.sugar_content !== null)
    .sort((a, b) => b.sugar_content! - a.sugar_content!) // 내림차순 정렬
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
  }, []);

  useEffect(() => {
    if (topN > results.length) {
      setTopN(results.length);
    }
  }, [results.length]);

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
    <View style={StyleSheet.absoluteFill} pointerEvents='box-none'>
      {/* topN 선택 드롭다운 */}
      {/* <TopNAppleSelector
        topN={topN}
        onChange={setTopN}
        maxN={Math.max(1, results.length)} // ✅ 최소 1개는 보장
      /> */}

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

      {/* <VisualBar results={results} onChangeMinSugar={setMinSugar} minSugar={minSugar} /> */}
      {/* 🔶 Skia 마스킹 캔버스 */}
      <Canvas style={StyleSheet.absoluteFill}>
        {/* 전체 어두운 레이어 */}
        <Group>
          <Rect
            x={0}
            y={0}
            width={screenSize.width}
            height={screenSize.height}
            color='rgba(0, 0, 0, 0.5)'
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

          const isHighlighted =
            filterMode === 'topN'
              ? topNIds.includes(result.id)
              : result.sugar_content !== undefined &&
                result.sugar_content !== null &&
                result.sugar_content >= minSugar;
          console.log(
            `[TopN Debug] id=${result.id}, 당도=${result.sugar_content}, isHighlighted=${isHighlighted}`
          );

          if (isHighlighted) {
            return (
              <Rect
                key={`mask-${index}`}
                x={screenBbox.x1}
                y={screenBbox.y1}
                width={screenWidth}
                height={screenHeight}
                color={
                  isHighlighted ? 'rgba(0, 0, 0, 0)' : 'rgba(0, 0, 0, 0.5)'
                }
                blendMode='clear' // 핵심! 이걸로 해당 영역만 비워줌
              />
            );
          } else {
            return null;
          }
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
            <Pressable
              onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                if (result.id !== undefined) {
                  handleApplePress(result.id);
                }
              }}
              style={{
                position: 'absolute',
                left: screenBbox.x1,
                top: screenBbox.y1,
                width: screenWidth,
                height: screenHeight,
                borderWidth: 2,
                borderColor: 'rgba(255,0,0,0.5)',
                backgroundColor: 'rgba(255,0,0,0.08)',
                zIndex: 100,
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
                ? require('../assets/images/explamation_apple.png')
                : require('../assets/images/question_apple.png')
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
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  selectedText: {
    color: '#000',
    fontWeight: 'bold',
  },
  infoButton: {
    position: 'absolute',
    bottom: 20,
    right: 5,
    zIndex: 1000,
    elevation: 10,
  },
  infoIcon: {
    width: 58,
    height: 68,
    resizeMode: 'contain',
  },
});
