import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import type { CameraView as CameraViewType, CameraCapturedPicture } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { toggleCameraFacing } from '../api/cameraApi';
import { initSocketConnection, sendFrame, onPrediction } from '@/app_logic/features/socket/socketEvents';

export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraViewType | null>(null);
  const intervalRef = useRef<number | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);

  // ✅ 소켓 연결 + 예측 결과 핸들러 등록
  useEffect(() => {
    initSocketConnection();

    onPrediction((data) => {
      setPrediction(data.brix);
    });

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // ✅ 카메라 전환 버튼 핸들러
  const handleToggleCameraFacing = () => {
    setFacing(toggleCameraFacing);
  };

  // ✅ 프레임 전송 시작 함수
  const startSendingFrames = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    intervalRef.current = window.setInterval(() => {
      (async () => {
        if (cameraRef.current) {
          try {
            const photo = await cameraRef.current.takePictureAsync({
              base64: true,
              quality: 0.3,
            });
            console.log(photo?.base64?.slice(0, 20))
    
            if (photo?.base64) {
              sendFrame({ image: photo.base64 });
            }
          } catch (err) {
            console.warn('📸 촬영 실패:', err);
          }
        }
      })();
    }, 1000);
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
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={handleToggleCameraFacing}>
            <Text style={styles.text}>카메라 전환</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.button} onPress={startSendingFrames}>
            <Text style={styles.text}>촬영 시작</Text>
          </TouchableOpacity>
        </View>

        {prediction !== null && (
          <View style={styles.predictionBox}>
            <Text style={styles.predictionText}>예측 당도: {prediction}</Text>
          </View>
        )}
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
    position: 'absolute',
    bottom: 0,
    width: '100%',
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
  predictionBox: {
    position: 'absolute',
    top: 40,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 10,
    borderRadius: 10,
  },
  predictionText: {
    fontSize: 20,
    color: 'white',
  },
});
