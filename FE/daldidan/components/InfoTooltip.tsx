import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  Animated,
  TouchableWithoutFeedback,
} from 'react-native';

interface Props {
  onDismiss: () => void;
}

export default function InfoTooltip({ onDismiss }: Props) {
  const opacity = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(10)).current;

  const [visible, setVisible] = useState(true); // fade-out 후 제거를 위한 제어

  useEffect(() => {
    // 등장 애니메이션
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(translateY, {
        toValue: 0,
        duration: 300,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const handleDismiss = () => {
    // 사라지는 애니메이션 후 onDismiss 호출
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: 0,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(translateY, {
        toValue: 10,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setVisible(false); // 내부 상태
      onDismiss(); // 부모 상태도 닫기
    });
  };

  if (!visible) return null;

  return (
    <TouchableWithoutFeedback onPress={handleDismiss}>
      <View style={StyleSheet.absoluteFill}>
        <Animated.View
          style={[
            styles.tooltipContainer,
            {
              opacity,
              transform: [{ translateY }],
            },
          ]}
        >
          <View style={styles.bubble}>
            <Text style={styles.text}>🍎 슬라이더로 당도를 설정해보세요!</Text>
            <Text style={styles.text}>🍎 사과를 터치하면 당도가 나와요!</Text>
          </View>
          <View style={styles.arrow} />
        </Animated.View>
      </View>
    </TouchableWithoutFeedback>
  );
}

const styles = StyleSheet.create({
  tooltipContainer: {
    position: 'absolute',
    bottom: 120,
    right: 20,
    zIndex: 999,
    alignItems: 'flex-end',
  },
  bubble: {
    backgroundColor: 'white',
    padding: 12,
    borderRadius: 8,
    maxWidth: 240,
    elevation: 5,
  },
  icon: {
    width: 40,
    height: 40,
    alignSelf: 'center',
    marginBottom: 6,
  },
  text: {
    fontSize: 11,
    marginBottom: 13,
  },
  arrow: {
    width: 0,
    height: 0,
    borderTopWidth: 10,
    borderTopColor: 'white',
    borderLeftWidth: 8,
    borderLeftColor: 'transparent',
    borderRightWidth: 8,
    borderRightColor: 'transparent',
    alignSelf: 'flex-end',
    marginRight: 16,
  },
});
