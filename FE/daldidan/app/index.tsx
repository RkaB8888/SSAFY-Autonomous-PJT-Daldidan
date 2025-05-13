import React from 'react';
import CameraViewNoDetect from '../components/CameraViewNoDetect';
import DebugFrameScreen from './debug-frame';

export default function App() {
  // 메인 / 근접 시 / 원거리 시 카메라 뷰 분기처리를 해야함
  return <DebugFrameScreen />;
}
