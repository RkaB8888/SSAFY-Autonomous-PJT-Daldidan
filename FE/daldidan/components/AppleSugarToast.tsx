import React, { useEffect, useRef, useState } from 'react';
import {
  Animated,
  Text,
  StyleSheet,
  View,
  LayoutChangeEvent,
} from 'react-native';

interface AppleSugarToastProps {
  visible: boolean;
  sugarContent: string | null;
  position: { x: number; y: number };
  onHide: () => void;
  onLayoutMeasured?: (width: number) => void;
}

export default function AppleSugarToast({
  visible,
  sugarContent,
  position,
  onHide,
  onLayoutMeasured,
}: AppleSugarToastProps) {
  const opacity = useRef(new Animated.Value(0)).current;
  const scale = useRef(new Animated.Value(0.7)).current;
  const [measured, setMeasured] = useState(false);

  useEffect(() => {
    if (visible) {
      opacity.setValue(0);
      scale.setValue(0.7);
      Animated.parallel([
        Animated.timing(opacity, {
          toValue: 1,
          duration: 120,
          useNativeDriver: true,
        }),
        Animated.spring(scale, {
          toValue: 1.15,
          friction: 4,
          useNativeDriver: true,
        }),
      ]).start(() => {
        Animated.spring(scale, {
          toValue: 1,
          friction: 5,
          useNativeDriver: true,
        }).start();
        setTimeout(() => {
          Animated.timing(opacity, {
            toValue: 0,
            duration: 220,
            useNativeDriver: true,
          }).start(onHide);
        }, 700);
      });
    }
  }, [visible]);

  const handleLayout = (e: LayoutChangeEvent) => {
    if (!measured && onLayoutMeasured) {
      setMeasured(true);
      onLayoutMeasured(e.nativeEvent.layout.width);
    }
  };

  if (!visible || !sugarContent) return null;

  return (
    <Animated.View
      style={[
        styles.toast,
        {
          left: position.x,
          top: position.y,
          opacity,
          transform: [{ scale }],
        },
      ]}
      onLayout={handleLayout}
    >
      <Text style={[styles.sugarText, { fontFamily: 'Maplestory' }]}>
        {sugarContent}
        <Text style={[styles.bx, { fontFamily: 'Maplestory' }]}>Bx</Text>
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  toast: {
    position: 'absolute',
    zIndex: 999,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#b71c1c',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.7,
    shadowRadius: 18,
    elevation: 18,
  },
  sugarText: {
    color: '#fffbe6',
    fontFamily: 'Maplestory',
    fontSize: 30,
    textShadowColor: '#b71c1c',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 8,
    letterSpacing: 1,
    includeFontPadding: false,
  },
  bx: {
    color: '#ffd600',
    fontFamily: 'Maplestory',
    fontWeight: 'bold',
    fontSize: 25,
    marginLeft: 2,
    textShadowColor: '#b71c1c',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
    includeFontPadding: false,
  },
});
