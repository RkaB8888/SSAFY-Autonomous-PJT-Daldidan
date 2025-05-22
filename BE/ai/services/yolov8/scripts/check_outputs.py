import tensorflow as tf

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(
    model_path="services/yolov8/models/yolov8n_bbox_int8.tflite"
)
interpreter.allocate_tensors()

# 출력 텐서 정보 출력
output_details = interpreter.get_output_details()

for i, detail in enumerate(output_details):
    print(f"Output {i}:")
    print(f"  name  : {detail['name']}")
    print(f"  shape : {detail['shape']}")
    print(f"  dtype : {detail['dtype']}")
    print()
