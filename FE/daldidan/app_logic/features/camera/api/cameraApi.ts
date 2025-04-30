import { CameraType } from 'expo-camera';

export const toggleCameraFacing = (currentFacing: CameraType): CameraType => {
  return currentFacing === 'back' ? 'front' : 'back';
};
