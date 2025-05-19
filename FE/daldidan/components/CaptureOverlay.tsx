// components/CaptureOverlay.tsx
import React, { useEffect, useState } from 'react';
import { View, Image, StyleSheet, Text, Dimensions } from 'react-native';
import LottieView from 'lottie-react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';

interface FramePair {
  character: any;
  camera: any;
}

interface CaptureOverlayProps {
  visible: boolean;
  framePair: FramePair;
}

interface Flash {
  id: number;
  top: number;
  left: number;
}

export default function CaptureOverlay({
  visible,
  framePair,
}: CaptureOverlayProps) {
  const [flashes, setFlashes] = useState<Flash[]>([]);
  const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } =
    Dimensions.get('window');
  const translateY = useSharedValue(0);
  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  useEffect(() => {
    if (!visible) {
      setFlashes([]);
      return;
    }

    const count = 6;
    const newFlashes: Flash[] = [];
    const timers: NodeJS.Timeout[] = [];

    for (let i = 0; i < count; i++) {
      const timer = setTimeout(() => {
        newFlashes.push({
          id: Date.now() + i,
          top: Math.random() * SCREEN_HEIGHT,
          left: Math.random() * SCREEN_WIDTH,
        });
        setFlashes((prev) => [...prev, newFlashes[newFlashes.length - 1]]);
      }, i * 200); // 0.2초 간격
      timers.push(timer);
    }

    return () => {
      timers.forEach(clearTimeout);
      setFlashes([]);
    };
  }, [visible]);

  if (!visible) return null;

  return (
    <View style={styles.overlay}>
      {flashes.map((flash) => (
        <LottieView
          key={flash.id}
          source={require('../assets/lottie/flash.json')}
          autoPlay
          loop={true}
          style={[
            styles.flash,
            {
              top: flash.top,
              left: flash.left,
              transform: [{ translateX: -50 }, { translateY: -50 }],
            },
          ]}
        />
      ))}

      <View style={styles.dualImageContainer}>
        <Animated.Image
          source={framePair.character}
          style={[styles.characterImage, animatedStyle]}
        />
        <Animated.Image
          source={framePair.camera}
          style={[styles.cameraImage, animatedStyle]}
        />
      </View>

      <Text style={styles.text}>사과를 찍고 있어요!</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 100,
  },
  dualImageContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 102,
  },
  characterImage: {
    width: 130,
    height: 130,
    marginRight: 5,
  },
  cameraImage: {
    width: 200,
    height: 200,
  },
  flash: {
    position: 'absolute',
    width: 100,
    height: 100,
    zIndex: 101,
  },
  text: {
    position: 'absolute',
    bottom: 100,
    color: 'white',
    fontSize: 25,
    fontFamily: 'Maplestory',
    textAlign: 'center',
    zIndex: 103,
  },
});
