export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  score?: number;
  class_id?: number;
  sugar_content?: number;
}

export interface DetectionResult {
  detection: Detection;
  imageData: string;
  result?: any;
  timestamp: number;
}

export interface CroppedImageData {
  data: number[];
  width: number;
  height: number;
}
