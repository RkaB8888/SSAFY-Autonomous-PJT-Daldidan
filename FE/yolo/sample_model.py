import numpy as np
import tensorflow as tf

# 모델 로딩
interpreter = tf.lite.Interpreter(model_path="../daldidan/assets/model2.tflite")
interpreter.allocate_tensors()

# 입력 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 더미 입력 (모델이 요구하는 shape: [1, 640, 640, 3])
dummy_input = np.random.rand(1, 640, 640, 3).astype(np.float32)

# 입력 텐서 설정
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

# 출력 텐서 얻기
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Output shape:", output_data.shape)
print("Sample output values:", output_data[0, :, :10])  # 10개 anchor만 보기
