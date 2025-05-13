import { Camera, CameraType, CameraCapturedPicture } from 'expo-camera';
import { useRef, useState, useEffect } from 'react';
import { View, Button, Image, StyleSheet, Text } from 'react-native';

export default function CaptureScreen() {
  const cameraRef = useRef<Camera | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync(); // ✅ 올바른 권한 요청 방식
      setHasPermission(status === 'granted');
    })();
  }, []);

  const takePhoto = async () => {
    if (cameraRef.current) {
      const photo: CameraCapturedPicture = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 1,
      });
      setImageUri(`data:image/jpeg;base64,${photo.base64}`);
    }
  };

  if (hasPermission === null) return <Text>카메라 권한 요청 중...</Text>;
  if (hasPermission === false) return <Text>카메라 권한이 거부되었습니다.</Text>;

  return (
    <View style={{ flex: 1 }}>
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={CameraType.back}
        ratio="16:9"
      />
      <Button title="📷 사진 찍기" onPress={takePhoto} />
      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.imagePreview} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  camera: { flex: 1 },
  imagePreview: { width: '100%', height: 300, resizeMode: 'contain' },
});
