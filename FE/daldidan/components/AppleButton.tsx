import React, { useEffect, useRef, useState } from 'react';
import { Animated, Image, TouchableOpacity, StyleSheet } from 'react-native';
import { Detection } from '../hooks/types/objectDetection';

interface Props {
  detections: Detection[];
  onPress: () => void;
}

export default function AppleButton({ detections, onPress }: Props) {
  const [showButton, setShowButton] = useState(false);
  const scaleAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const hasApple = detections.some((d) => d.class_id === 52);
    setShowButton(hasApple);
  }, [detections]);

  useEffect(() => {
    if (showButton) {
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 4,
        useNativeDriver: true,
      }).start();
    } else {
      scaleAnim.setValue(0);
    }
  }, [showButton]);

  if (!showButton) return null;

  return (
    <Animated.View style={[styles.container, { transform: [{ scale: scaleAnim }] }]}>
      <TouchableOpacity onPress={onPress}>
        <Image source={require('../assets/apple.png')} style={styles.appleIcon} />
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 100,
    alignSelf: 'center',
    zIndex: 20,
  },
  appleIcon: {
    width: 60,
    height: 60,
  },
});
