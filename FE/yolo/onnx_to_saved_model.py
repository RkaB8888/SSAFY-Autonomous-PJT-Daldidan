# 변환 코드
import onnx
from onnx_tf.backend import prepare

# ONNX 모델 로드
onnx_model = onnx.load("./runs/detect/apple_detector_v214/weights/best.onnx")

# TensorFlow 모델로 변환
tf_rep = prepare(onnx_model)
tf_rep.export_graph("saved_model")
