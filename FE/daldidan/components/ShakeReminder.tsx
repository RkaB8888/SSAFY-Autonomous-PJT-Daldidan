// ShakeReminder.tsx (새 컴포넌트로 분리 추천)
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, Image, StyleSheet, Animated, Easing } from 'react-native';

export default function ShakeReminder() {
  const [visible, setVisible] = useState(false);
  const shakeAnim = useRef(new Animated.Value(0)).current;

  // 좌우 흔들기 애니메이션
  const startShake = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(shakeAnim, {
          toValue: -5,
          duration: 120,
          useNativeDriver: true,
          easing: Easing.linear,
        }),
        Animated.timing(shakeAnim, {
          toValue: 5,
          duration: 120,
          useNativeDriver: true,
          easing: Easing.linear,
        }),
        Animated.timing(shakeAnim, {
          toValue: 0,
          duration: 120,
          useNativeDriver: true,
          easing: Easing.linear,
        }),
      ])
    ).start();
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setVisible(true);
      startShake();

      setTimeout(() => {
        setVisible(false);
      }, 3000); // 2초간 보여주고 숨기기
    }, 5000); // 5초마다

    return () => clearInterval(interval);
  }, []);

  if (!visible) return null;

  return (
    <View style={styles.container}>
      <Animated.Image
        source={require('../assets/images/phone_icon.png')}
        style={[styles.image, { transform: [{ translateX: shakeAnim }] }]}
      />
      <Text style={styles.text}>흔들어서 재측정하기</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  image: {
    width: 60,
    height: 60,
    resizeMode: 'contain',
  },
  text: {
    color: '#fff',
    marginTop: 6,
    fontSize: 14,
    fontWeight: 'bold',
  },
});
