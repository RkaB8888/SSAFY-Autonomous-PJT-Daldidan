# services/yolov8/scripts/check_sigmoid.py
from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "C:/Users/SSAFY/Desktop/SSAFY-Autonomous-PJT/S12P31E206/BE/ai/services/yolov8/models/yolov8n_seg_float32.tflite"

interp = Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
ops = interp._get_ops_details()

# 1) 반환된 dict의 키를 한번 찍어보고
print("첫 번째 op의 키 목록:", ops[0].keys())

# 2) op_name이 'LOGISTIC'인 연산이 있는지 확인
has_logistic = any(op.get("op_name", "").upper() == "LOGISTIC" for op in ops)
print("모델에 LOGISTIC(Sigmoid) 연산 포함 여부:", has_logistic)

# (추가) 모든 op_name을 보고 싶으면
print("모든 op_name들:", [op.get("op_name") for op in ops])
