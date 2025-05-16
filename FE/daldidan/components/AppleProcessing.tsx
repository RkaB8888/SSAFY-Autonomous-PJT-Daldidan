// AppleProcessing.tsx
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import LottieView from 'lottie-react-native';

interface Props {
  status: 'juicing';
}

export default function AppleProcessing({ status }: Props) {
  const getAnimation = () => {
    // if (status === 'peeling') return require('../assets/animations/apple_peeling.json');
    if (status === 'juicing') return require('../assets/lottie/apple_juicy.json');
  };

  const getText = () => {
    // if (status === 'peeling') return '사과를 깎는 중...';
    if (status === 'juicing') return '사과즙 짜는 중...';
  };

  return (
    <View style={styles.overlay}>
      <LottieView
        source={getAnimation()}
        autoPlay
        loop
        style={styles.animation}
      />
      <Text style={styles.text}>{getText()}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.5)',
    zIndex: 999,
  },
  animation: {
    width: 200,
    height: 200,
  },
  text: {
    marginTop: 20,
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
  },
});
