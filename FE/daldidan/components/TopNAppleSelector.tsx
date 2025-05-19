// components/TopNAppleSelector.tsx
import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Picker } from '@react-native-picker/picker';

interface Props {
  topN: number;
  onChange: (n: number) => void;
  maxN: number;
}

export default function TopNAppleSelector({ topN, onChange, maxN }: Props) {
  return (
    <View style={styles.dropdownWrapper}>
      <Picker
        selectedValue={topN}
        style={[styles.picker, { fontFamily: 'Maplestory' }]}
        onValueChange={(value) => onChange(value)}
      >
        {Array.from({ length: maxN }, (_, i) => i + 1).map((n) => (
          <Picker.Item
            key={n}
            label={`Top ${n}`}
            value={n}
            style={{ fontFamily: 'Maplestory' }}
          />
        ))}
      </Picker>
    </View>
  );
}

const styles = StyleSheet.create({
  dropdownWrapper: {
    position: 'absolute',
    top: 100,
    left: 10,
    zIndex: 100,
    backgroundColor: 'white',
    borderRadius: 4,
  },
  picker: {
    width: 120,
    height: 50,
    paddingTop: -4,
    paddingBottom: -4,
  },
});
