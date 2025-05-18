// components/TopNAppleSelector.tsx
import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Picker } from '@react-native-picker/picker';

interface Props {
  topN: number;
  onChange: (n: number) => void;
}

export default function TopNAppleSelector({ topN, onChange }: Props) {
  return (
    <View style={styles.dropdownWrapper}>
      <Picker
        selectedValue={topN}
        style={styles.picker}
        onValueChange={(value) => onChange(value)}
      >
        {[1, 2, 3, 4, 5].map((n) => (
          <Picker.Item key={n} label={`Top ${n}`} value={n} />
        ))}
      </Picker>
    </View>
  );
}

const styles = StyleSheet.create({
  dropdownWrapper: {
    position: 'absolute',
    top: 10,
    left: 10,
    zIndex: 100,
    backgroundColor: 'white',
    borderRadius: 4,
  },
  picker: {
    width: 120,
    height: 44,
  },
});
