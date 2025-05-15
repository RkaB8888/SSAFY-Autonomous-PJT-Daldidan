// AppleBar.tsx
import React from 'react';
import { View, StyleSheet, Text } from 'react-native';
import { Detection } from '../hooks/types/objectDetection';

interface Props {
  detections: Detection[];
}

export default function AppleBar({ detections }: Props) {
  const appleCount = detections.filter((d) => d.class_id === 52).length;
  const maxApples = 5;

  return (
    <View style={styles.wrapper}>
      {/* 기본 바 배경 */}
      <View style={styles.bar} />

      {/* 사과 이모지 겹쳐서 표시 */}
      {Array.from({ length: Math.min(appleCount, maxApples) }).map((_, index) => (
        <Text key={index} style={[styles.emoji, { left: index * 45 }]}>
          🍎
        </Text>
      ))}

      {/* 추가 사과가 많을 때 +n 표시 */}
      {appleCount > maxApples && (
        <Text style={[styles.extra, { left: maxApples * 45 }]}>+{appleCount - maxApples}</Text>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    position: 'absolute',
    top: 40,
    alignSelf: 'center',
    height: 50,
    justifyContent: 'center',
    zIndex: 999,
  },
  bar: {
    width: 300,
    height: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 20,
  },
  emoji: {
    position: 'absolute',
    fontSize: 30,
    top: 4,
  },
  extra: {
    position: 'absolute',
    top: 10,
    fontSize: 16,
    color: 'white',
  },
});
