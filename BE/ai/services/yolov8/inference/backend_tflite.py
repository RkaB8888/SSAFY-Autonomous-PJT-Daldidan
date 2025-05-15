# services/yolov8/inference/backend_tflite.py

import numpy as np
import tensorflow as tf  # TFLite 인터프리터는 tensorflow 패키지에 포함됨


class TFLiteYoloV8Backend:
    def __init__(self, model_path: str):
        # TFLite 모델 로딩
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # 입력 / 출력 텐서 정보 저장
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_dtype = self.input_details[0]["dtype"]

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Args:
            input_tensor (np.ndarray): [1, H, W, 3] float32 normalized tensor
        Returns:
            np.ndarray: YOLOv8 TFLite raw output tensor
        """
        # 입력 텐서 설정
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)

        # 추론 실행
        self.interpreter.invoke()

        # 출력 결과 추출
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output_data
