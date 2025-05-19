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
  const barHeightRef = useRef(1);
  const barTopRef = useRef(0);

  const FIXED_MIN_BRIX = 8;
  const FIXED_MAX_BRIX = 18;
  const BAR_HEIGHT = 300;

  // 🍎 사과 핸들용 PanResponder
  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: (_, gestureState) => {
        const y = gestureState.y0 - barTopRef.current;
        const ratio = 1 - Math.max(0, Math.min(1, y / barHeightRef.current));
        const sugarValue = FIXED_MIN_BRIX + ratio * (FIXED_MAX_BRIX - FIXED_MIN_BRIX);
        const rounded = parseFloat(sugarValue.toFixed(1));
        onChangeMinSugar?.(rounded);
        return true;
      },
      onPanResponderMove: (_, gestureState) => {
        const y = gestureState.moveY - barTopRef.current;
        const ratio = 1 - Math.max(0, Math.min(1, y / barHeightRef.current));
        const sugarValue = FIXED_MIN_BRIX + ratio * (FIXED_MAX_BRIX - FIXED_MIN_BRIX);
        const rounded = parseFloat(sugarValue.toFixed(1));
        onChangeMinSugar?.(rounded);
      },
    })
  ).current;

  const appleTop = ((1 - (minSugar - FIXED_MIN_BRIX) / (FIXED_MAX_BRIX - FIXED_MIN_BRIX)) * BAR_HEIGHT) - 17.5;

  return (
    <View style={styles.container}>
      <View
        ref={barRef}
        style={{ height: BAR_HEIGHT, width: 20, justifyContent: 'center' }}
        onLayout={() => {
          barRef.current?.measureInWindow((x, y, width, height) => {
            barTopRef.current = y;
            barHeightRef.current = height;
          });
        }}
      >
        <LinearGradient
          colors={['#a8e063', '#ff5f6d']}
          start={{ x: 0, y: 0 }}
          end={{ x: 0, y: 1 }}
          style={{ height: '100%', width: '100%', borderRadius: 10 }}
        />

        {/* 🔥 사과 핸들 드래그 가능 */}
        <View
          {...panResponder.panHandlers}
          style={{
            position: 'absolute',
            top: appleTop,
            left: -7.5,
            zIndex: 10,
          }}
        >
          <Image
            source={appleImage}
            style={{ width: 35, height: 35 }}
            resizeMode="contain"
          />
        </View>

        {/* 실시간 최소 당도 표시 */}
        <View
          style={{
            position: 'absolute',
            top: appleTop - 25,
            left: -50,
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
});
