import { LinearGradient } from 'expo-linear-gradient';
import React, { useState } from 'react';
import {
  Animated,
  Image,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';

interface VisualBarProps {
  results: AnalyzedObjectResult[];
  onApplePress?: (appleId: string | number) => void;
}

const tastyImage = require('../assets/images/tasty.png');
const notTastyImage = require('../assets/images/nottasty.png');

export default function VisualBar({ results, onApplePress }: VisualBarProps) {
  const [selectedAppleId, setSelectedAppleId] = useState<string | number | null>(null);
  const [glowAnim] = useState(new Animated.Value(0));

  const BAR_HEIGHT = 15;

  const brixValues = results.map(r => r.sugar_content ?? 0);
  const minBrix = Math.floor(Math.min(...brixValues));
  const maxBrix = Math.ceil(Math.max(...brixValues));
  const midBrix = (minBrix + maxBrix) / 2;

  const getAppleXPercent = (sugar: number) => {
    const ratio = (sugar - minBrix) / (maxBrix - minBrix || 1);
    return `${(ratio * 100).toFixed(1)}%`;
  };

  const handleApplePress = (appleId: string | number) => {
    setSelectedAppleId(appleId);
    onApplePress?.(appleId);

    Animated.sequence([
      Animated.timing(glowAnim, {
        toValue: 1,
        duration: 500,
        useNativeDriver: true,
      }),
      Animated.timing(glowAnim, {
        toValue: 0,
        duration: 500,
        useNativeDriver: true,
      }),
    ]).start();
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#f7e36a', '#b4e06c']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={[
          styles.bar,
          {
            zIndex: 1,
            top: '250%',
            height: BAR_HEIGHT,
            width: '90%',
            position: 'relative',
          },
        ]}
      >
        {results
          .sort((a, b) => (a.sugar_content ?? 0) - (b.sugar_content ?? 0))
          .map((result) => {
            const sugar = result.sugar_content ?? 0;
            const imageSource = sugar < midBrix ? tastyImage : notTastyImage;

            return (
              <TouchableOpacity
                key={result.id}
                onPress={() => handleApplePress(result.id)}
                style={{
                  position: 'absolute',
                  left: getAppleXPercent(sugar),
                  top: '-90%',
                  zIndex: 10,
                  marginLeft: -14, // 이미지 너비 절반만큼 왼쪽 정렬 보정
                }}
              >
                <Animated.View
                  style={{
                    transform: [
                      {
                        scale: glowAnim.interpolate({
                          inputRange: [0, 1],
                          outputRange: [1, 1.2],
                        }),
                      },
                    ],
                    opacity: glowAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [1, 0.8],
                    }),
                  }}
                >
                  <Image
                    source={imageSource}
                    style={{ width: 35, height: 35 }}
                    resizeMode="contain"
                  />
                </Animated.View>
              </TouchableOpacity>
            );
          })}
      </LinearGradient>

      <View style={{ position: 'absolute', left: 5, top: 25, zIndex: 1 }}>
        <Text style={{ fontSize: 24 }}>🍯</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    marginVertical: 12,
    width: '100%',
    paddingHorizontal: 20,
  },
  bar: {
    borderRadius: 14,
    justifyContent: 'center',
    position: 'relative',
  },
});
