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
            <Text style={styles.maintext}>Tips !</Text>
            <Text style={styles.text}>🍎 <Text style={styles.boldtext}>슬라이더</Text>로 <Text style={styles.boldtext}>최소당도</Text>를 설정해보세요!</Text>
            <Text style={styles.text}>🍎 <Text style={styles.boldtext}>사과를 터치</Text>하면 당도가 나와요!</Text>
            <Text style={styles.text}>
              🍎 가장 맛있는 <Text style={styles.boldtext}>3개의 사과</Text>만 볼 수 있어요!
            </Text>
            <Text style={styles.text}>
              🍎 <Text style={styles.boldtext}>14 brix</Text> 이상의 맛있는 사과를 골라봐요!
            </Text>
            <Text style={styles.text}>
              🍎 당도 정보는 참고용입니다. 정확하지 않을 수 있어요 !
            </Text>
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
    bottom: 140,
    right: 20,
    zIndex: 999,
    elevation: 10,
    alignItems: 'flex-end',
  },
  bubble: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 8,
    maxWidth: 350,
    zIndex: 999,
    elevation: 5,
  },
  icon: {
    width: 40,
    height: 40,
    alignSelf: 'center',
    marginBottom: 6,
  },
  maintext: {
    fontSize: 18,
    fontFamily: 'Maplestory',
    marginBottom: 13,
    color: 'orange',
  },
  text: {
    fontSize: 13,
    marginBottom: 13,
    fontFamily: 'Maplestory_Light',
  },
  boldtext: {
    fontSize: 13,
    fontFamily: 'Maplestory',
    marginBottom: 13,
    color: 'orange',
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
