import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import LottieView from 'lottie-react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';

export default function AppleHint() {
  const translateY = useSharedValue(0);

  useEffect(() => {
    const screenHeight = -400; // 임의 높이, 나중에 Dimensions로 처리 가능
    translateY.value = withRepeat(
      withSequence(
        withTiming(-screenHeight, { duration: 3000 }),
        withTiming(0, { duration: 3000 })
      ),
      -1,
      true
    );
  }, []);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  return (
    <View style={styles.container}>
      <Animated.View style={[StyleSheet.absoluteFillObject, animatedStyle]}>
        <LottieView
          source={require('../assets/lottie/scan.json')}
          autoPlay
          loop
          style={styles.lottie}
        />
      </Animated.View>
      <View style={styles.overlay}>
        <Text style={styles.subtitle}>🍎 사과를 비춰주세요 🍎</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 999,
  },
  lottie: {
    width: '100%',
    height: 300, // 스캔 바 높이
  },
  overlay: {
    position: 'absolute',
    bottom: '15%',
    alignItems: 'center',
  },
  subtitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingHorizontal: 20,
  },
});
