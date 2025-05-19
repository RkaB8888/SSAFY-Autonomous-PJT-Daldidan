import React, { useRef } from 'react';
import { StyleSheet } from 'react-native';
import LottieView from 'lottie-react-native';

interface Props {
  path?: any; // path를 선택적으로 변경
  color: string;
  onAnimationEnd?: () => void;
  position?: { x: number; y: number };
  size?: number; // 애니메이션 크기 prop 추가
}

export default function AppleJuiceAnimation({
  path,
  color,
  onAnimationEnd,
  position,
  size = 100, // 기본값 100으로 설정
}: Props) {
  const animationRef = useRef<LottieView>(null);

  return (
    <LottieView
      ref={animationRef}
      source={require('../assets/lottie/apple_pop.json')}
      style={[
        StyleSheet.absoluteFill,
        position && {
          position: 'absolute',
          left: position.x,
          top: position.y,
          width: size,
          height: size,
          transform: [{ translateX: -size / 2 }, { translateY: -size / 2 }], // 크기에 따라 중앙 정렬 오프셋 조정
        },
      ]}
      autoPlay={true}
      loop={false}
      onAnimationFinish={onAnimationEnd}
      speed={1}
    />
  );
}
