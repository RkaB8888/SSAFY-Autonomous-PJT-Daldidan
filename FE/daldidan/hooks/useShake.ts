import { Accelerometer } from 'expo-sensors';
import { useEffect } from 'react';

export function useShake(onShake: () => void, threshold = 1.5, interval = 1000) {
  useEffect(() => {
    let lastShakeTime = 0;

    const subscription = Accelerometer.addListener(({ x, y, z }) => {
      const acceleration = Math.sqrt(x * x + y * y + z * z);
      const now = Date.now();
      if (acceleration > threshold && now - lastShakeTime > interval) {
        lastShakeTime = now;
        onShake();
      }
    });

    Accelerometer.setUpdateInterval(100);

    return () => subscription.remove();
  }, [onShake, threshold, interval]);
}
