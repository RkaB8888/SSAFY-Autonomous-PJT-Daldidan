import numpy as np
import tensorflow as tf
import cv2

# 1. 사과 이미지 로드
image = cv2.imread('./images/val/apple_1.jpg')
image = cv2.resize(image, (448, 448))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_data = image_rgb.astype(np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)  # [1, 640, 640, 3]

# 2. 모델 로딩
interpreter = tf.lite.Interpreter(model_path='apple_detector.tflite')
interpreter.allocate_tensors()

# 3. 추론 실행
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])  # [1, 300, 6]

# 4. 감지 결과 필터링
for det in output[0]:
    x1, y1, x2, y2, score, cls = det
    if score > 0.3:
        print(f"📦 box: ({x1:.2f}, {y1:.2f}) → ({x2:.2f}, {y2:.2f}), score: {score:.3f}, class: {int(cls)}")
