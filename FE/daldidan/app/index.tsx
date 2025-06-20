import React from 'react';
import CameraViewNoDetect from '../components/CameraViewNoDetect';
import { InfoTooltipProvider } from '../components/InfoTooltipContext';
import SliderTest from '@/components/SliderTest';

export default function App() {
  // 메인 / 근접 시 / 원거리 시 카메라 뷰 분기처리를 해야함
  return (
    <InfoTooltipProvider>
      <CameraViewNoDetect />
    </InfoTooltipProvider>
    // <SliderTest />
  ) 
}
