import React, { useState } from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import Slider from '@react-native-community/slider';

export default function SliderTest() {
  const [val, setVal] = useState(10);

  return (
    <View style={styles.container}>
      <Text style={styles.valueText}>{val.toFixed(1)} Bx</Text>
<Slider
  style={styles.slider}
  minimumValue={8}
  maximumValue={18}
  step={0.1}
  value={val}
  onValueChange={setVal}
  minimumTrackTintColor="#ff5f6d"
  maximumTrackTintColor="#a8e063"
  // thumbImage={require('../assets/apple.png')} ← 주석 처리!
  thumbTintColor="#ff5f6d"
/>

    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 60,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  slider: {
    width: '100%',
    height: 40,
  },
  valueText: {
    color: 'white',
    marginBottom: 20,
    fontSize: 20,
  },
});
