// AppleHint.tsx
import React, { useEffect } from 'react';
import { View, Text, Image, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';

const phoneImage = require('../assets/images/phone_icon.png'); // 폰 그림 경로로 수정

export default function AppleHint() {
  const rotation = useSharedValue(0);

  useEffect(() => {
    rotation.value = withRepeat(
      withSequence(
        withTiming(-10, { duration: 150 }),
        withTiming(10, { duration: 150 }),
        withTiming(-10, { duration: 150 }),
        withTiming(0, { duration: 150 })
      ),
      -1, // 무한 반복
      false
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ rotateZ: `${rotation.value}deg` }],
  }));

  return (
    <View style={styles.container}>
      <Animated.Image
        source={phoneImage}
        style={[styles.image, animatedStyle]}
        resizeMode="contain"
      />
      {/* <Text style={styles.title}>인식된 사과가 없습니다!</Text> */}
      <Text style={styles.subtitle}>🍎사과를 비춰주세요🍎</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.4)',
    zIndex: 999,
  },
  image: {
    width: 120,
    height: 120,
    marginBottom: 20,
  },
  title: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
  subtitle: {
    color: 'white',
    fontSize: 16,
    marginTop: 4,
  },
});
