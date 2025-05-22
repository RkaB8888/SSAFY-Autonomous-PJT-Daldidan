import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  PanResponder,
  GestureResponderEvent,
  PanResponderGestureState,
} from 'react-native';
import { AnalyzedObjectResult } from '../hooks/types/objectDetection';
import { LinearGradient } from 'expo-linear-gradient';


interface VisualBarProps {
  results: AnalyzedObjectResult[];
  onChangeMinSugar?: (sugar: number) => void;
  minSugar: number;
}


const appleUnsweet = require('../assets/images/emot_unsweet.png');
const appleSoso = require('../assets/images/emot_soso.png');
const appleSweet = require('../assets/images/emot_sweet.png');

const FIXED_MIN_BRIX = 8;
const FIXED_MAX_BRIX = 18;
const SLIDER_HEIGHT = 300;
const THUMB_SIZE = 50;

export default function VisualBar({ results, onChangeMinSugar, minSugar }: VisualBarProps) {
  const [internalSugar, setInternalSugar] = useState(minSugar);

  const handlePan = useCallback((gestureState: PanResponderGestureState) => {
    const clampedY = Math.max(0, Math.min(SLIDER_HEIGHT, gestureState.dy + SLIDER_HEIGHT * (1 - (minSugar - FIXED_MIN_BRIX) / (FIXED_MAX_BRIX - FIXED_MIN_BRIX))));
    const ratio = 1 - clampedY / SLIDER_HEIGHT;
    const newSugar = Math.round((FIXED_MIN_BRIX + (FIXED_MAX_BRIX - FIXED_MIN_BRIX) * ratio) * 10) / 10;
    setInternalSugar(newSugar);
    onChangeMinSugar?.(newSugar);
  }, [minSugar]);

  const getAppleImageBySugar = (sugar: number) => {
  if (sugar < 11) return appleUnsweet;
  if (sugar < 14) return appleSoso;
  return appleSweet;
};

  const panResponder = PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    onPanResponderMove: (_, gestureState) => handlePan(gestureState),
    onPanResponderGrant: (_, gestureState) => handlePan(gestureState),
    onPanResponderRelease: (_, gestureState) => handlePan(gestureState),
  });

  const ratio = (internalSugar - FIXED_MIN_BRIX) / (FIXED_MAX_BRIX - FIXED_MIN_BRIX);
  const topPosition = (1 - ratio) * SLIDER_HEIGHT - THUMB_SIZE / 2;

  return (
    <View style={styles.container}>
      <Text style={styles.sugarText}>{internalSugar.toFixed(1)} Bx</Text>

      <View style={styles.sliderTrack}>
        <LinearGradient
          colors={['#ff5f6d', '#a8e063']} // ← 원하는 색상 조합
          style={styles.sliderBar}
          start={{ x: 0.5, y: 0 }}
          end={{ x: 0.5, y: 1 }}
        />
        <View
          style={[styles.thumb, { top: topPosition }]}
          {...panResponder.panHandlers}
        >
          <Image source={getAppleImageBySugar(internalSugar)} style={styles.appleImage} />
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    height: SLIDER_HEIGHT + 55,
    justifyContent: 'center',
    // paddingHorizontal: 20,
  },
  sugarText: {
    position: 'absolute',
    fontFamily: 'Maplestory_Light',
    backgroundColor: 'rgba(242, 105, 105, 0.7)',
    bottom : -15,
    color: 'white',
    fontSize: 12,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    width: 75,
    left: -8, 
    textAlign: 'center',
    textAlignVertical: 'center'
  },
  sliderTrack: {
    height: SLIDER_HEIGHT,
    width: 40,
    backgroundColor: 'transparent',
    justifyContent: 'flex-start',
    alignItems: 'center',
    position: 'relative',
  },
  sliderBar: {
    position: 'absolute',
    width: 4,
    backgroundColor: '#ff5f6d',
    top: 0,
    bottom: 0,
    left: '50%',
    transform: [{ translateX: -2 }],
    borderRadius: 2,
  },
  thumb: {
    position: 'absolute',
    width: THUMB_SIZE,
    height: THUMB_SIZE,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  appleImage: {
    width: THUMB_SIZE,
    height: THUMB_SIZE,
    resizeMode: 'contain',
  },
});
