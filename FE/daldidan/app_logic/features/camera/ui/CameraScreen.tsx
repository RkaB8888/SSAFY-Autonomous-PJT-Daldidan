import React from 'react';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import type { CameraView as CameraViewType } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { toggleCameraFacing } from '../api/cameraApi';
import { initSocketConnection, sendFrame, onPrediction } from '@/app_logic/features/socket/socketEvents';
import { PredictionData, ApplePrediction } from '@/app_logic/features/socket/types';
import Svg, { Rect, Text as SvgText } from 'react-native-svg';

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraViewType | null>(null);
  const intervalRef = useRef<number | null>(null);
  const [predictions, setPredictions] = useState<ApplePrediction[]>([]);
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    initSocketConnection();
    onPrediction((data: PredictionData) => setPredictions(data.results));

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const handleToggleCameraFacing = () => setFacing(toggleCameraFacing);

  const startSendingFrames = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = window.setInterval(() => {
      (async () => {
        if (cameraRef.current) {
          try {
            const photo = await cameraRef.current.takePictureAsync({
              base64: true,
              quality: 0.3,
            });

            if (photo?.base64) sendFrame({ image: photo.base64 });
          } catch (err) {
            console.warn('📸 촬영 실패:', err);
          }
        }
      })();
    }, 3000);

    setIsSending(true);
  };

  const stopSendingFrames = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    setIsSending(false);
  };

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>카메라 권한이 필요합니다.</Text>
        <Button onPress={requestPermission} title="권한 요청" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing={facing}>
        <Svg style={StyleSheet.absoluteFill}>
          {predictions.map((apple) => (
            <React.Fragment key={apple.id}>
              <Rect
                x={apple.box[0]}
                y={apple.box[1]}
                width={apple.box[2]}
                height={apple.box[3]}
                stroke="red"
                strokeWidth={2}
                fill="transparent"
              />
              <SvgText
                x={apple.box[0] + 5}
                y={apple.box[1] + 20}
                fill="white"
                fontSize="14"
              >
                🍎 {apple.brix} °Brix
              </SvgText>
            </React.Fragment>
          ))}
        </Svg>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={handleToggleCameraFacing}>
            <Text style={styles.text}>카메라 전환</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.button}
            onPress={isSending ? stopSendingFrames : startSendingFrames}
          >
            <Text style={styles.text}>{isSending ? '촬영 중지' : '촬영 시작'}</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  camera: { flex: 1 },
  buttonContainer: {
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 54,
    position: 'absolute',
    bottom: 40,
    width: '80%',
    justifyContent: 'space-around',
  },
  button: {
    backgroundColor: 'rgba(0,0,0,0.4)',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 10,
  },
  text: {
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
});
