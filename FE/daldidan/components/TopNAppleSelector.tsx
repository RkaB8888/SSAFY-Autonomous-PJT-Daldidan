import React, { useState } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';

const TopNAppleSelector = ({ topN, onChange, maxN }: {
  topN: number;
  onChange: (n: number) => void;
  maxN: number;
}) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const options = Array.from({ length: maxN }, (_, i) => i + 1);

  const handleSelect = (n: number) => {
    onChange(n);
    setIsDropdownOpen(false);
  };

  return (
    <View style={{ marginBottom: 10, zIndex: 9999 }}>
      <Pressable
        onPress={() => setIsDropdownOpen(prev => !prev)}
        style={styles.button}
      >
        <Text style={[styles.buttonText]}>Top {topN}개 보기 ▼</Text>
      </Pressable>

      {isDropdownOpen && (
        <View style={styles.dropdown}>
          {options.map((n) => (
            <Pressable key={n} onPress={() => handleSelect(n)} style={styles.option}>
              <Text style={styles.optionText}>{n}개</Text>
            </Pressable>
          ))}
        </View>
      )}
    </View>
  );
};

export default TopNAppleSelector;

const styles = StyleSheet.create({
  button: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#ff8c00',
    borderRadius: 6,
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  dropdown: {
    marginTop: 5,
    position: 'absolute',
    backgroundColor: 'white',
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 4,
    elevation: 10,
    width: 100,
  },
  option: {
    padding: 10,
  },
  optionText: {
    fontSize: 14,
  },
});
