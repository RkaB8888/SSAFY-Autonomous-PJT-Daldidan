import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import type {CameraView as CameraViewType } from 'expo-camera'
import { useEffect, useRef, useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { toggleCameraFacing } from '../api/cameraApi';
import { initSocketConnection, sendFrame, onPrediction } from '@/app_logic/features/socket/socketEvents';


export default function CameraScreen() {
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraViewType | null>(null); // 카메라 인스턴스 관리
  const intervalRef = useRef<number  | null>(null); // 프레임 전송 주기 관리
  const [prediction, setPrediction] = useState<number | null>(null); // 예측 결과 상태

  // 소켓 연결 및 예측 수신
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
  
  

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>카메라 권한이 필요합니다.</Text>
        <Button onPress={requestPermission} title='권한 요청' />
      </View>
    );
  }

  const handleToggleCameraFacing = () => {
    setFacing(toggleCameraFacing);
  };


  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.button}
            onPress={handleToggleCameraFacing}
          >
            <Text style={styles.text}>카메라 전환</Text>
          </TouchableOpacity>
        </View>
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
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
});
