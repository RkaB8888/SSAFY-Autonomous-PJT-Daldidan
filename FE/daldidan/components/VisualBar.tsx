import { LinearGradient } from 'expo-linear-gradient';
import React, { useRef } from 'react';
import {
  Image,
  PanResponder,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';

interface VisualBarProps {
  results: AnalyzedObjectResult[];
  onChangeMinSugar?: (sugar: number) => void;
  minSugar: number;
}

const appleImage = require('../assets/apple.png');

export default function VisualBar({ results, onChangeMinSugar, minSugar }: VisualBarProps) {
  const barRef = useRef<View>(null);
  const barWidthRef = useRef(1);
  const barLeftRef = useRef(0);

  const FIXED_MIN_BRIX = 8;
  const FIXED_MAX_BRIX = 18;
  const BAR_HEIGHT = 15;

  // 🍎 사과 핸들용 PanResponder
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onPanResponderMove: (e, gestureState) => {
        const x = gestureState.moveX - barLeftRef.current;
        const ratio = Math.max(0, Math.min(1, x / barWidthRef.current));
        const sugarValue = FIXED_MIN_BRIX + ratio * (FIXED_MAX_BRIX - FIXED_MIN_BRIX);
        const rounded = parseFloat(sugarValue.toFixed(1));
        console.log('🍯 최소 당도 조절됨 →', rounded);
        onChangeMinSugar?.(rounded);
      },
    })
  ).current;

  return (
    <View style={styles.container}>
      <View
        ref={barRef}
        style={{ height: 10, width: '100%', marginLeft: '10%' }} // 🔥 최소한의 높이와 너비 지정!
        onLayout={(e) => {
          barWidthRef.current = e.nativeEvent.layout.width;
          barLeftRef.current = e.nativeEvent.layout.x;
        }}
      >
        <LinearGradient
          colors={['#a8e063', '#ff5f6d']} // 왼쪽 연두 → 오른쪽 빨간
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={[
            styles.bar,
            {
              zIndex: 1,
              top: '250%',
              height: BAR_HEIGHT,
              width: '90%',
              position: 'relative',
            },
          ]}
        >
          {/* 실시간 최소 당도 표시 */}
          <View
            style={{
              position: 'absolute',
              left: `${((minSugar - FIXED_MIN_BRIX) / (FIXED_MAX_BRIX - FIXED_MIN_BRIX)) * 100}%`,
              top: -25, // 사과 위에 표시되도록
              transform: [{ translateX: -30 }],
              zIndex: 20,
            }}
            pointerEvents="none"
          >
            <Text
              style={{
                backgroundColor: 'rgba(0,0,0,0.7)',
                color: 'white',
                fontSize: 12,
                paddingHorizontal: 6,
                paddingVertical: 2,
                borderRadius: 4,
              }}
            >
              {minSugar.toFixed(1)} Bx
            </Text>
          </View>

          {/* 🔥 사과 핸들 드래그 가능 */}
          <View
            {...panResponder.panHandlers}
            style={{
              position: 'absolute',
              left: `${((minSugar - FIXED_MIN_BRIX) / (FIXED_MAX_BRIX - FIXED_MIN_BRIX)) * 100}%`,
              // top: 5,
              zIndex: 10,
              marginLeft: -17,
            }}
          >
            <Image
              source={appleImage}
              style={{ width: 35, height: 35 }}
              resizeMode="contain"
            />
          </View>
        </LinearGradient>
      </View>

      <View style={{ position: 'absolute', left: 5, top: 10, zIndex: 1 }}>
        <Text style={{ fontSize: 24 }}>🍯</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    marginVertical: 12,
    width: '100%',
    paddingHorizontal: 20,
  },
  bar: {
    borderRadius: 14,
    justifyContent: 'center',
    position: 'relative',
  },
});
