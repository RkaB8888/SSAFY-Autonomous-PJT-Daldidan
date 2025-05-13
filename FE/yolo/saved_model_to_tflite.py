import tensorflow as tf

# 변환기 생성
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

# ✅ 핵심 옵션: Flex ops 허용
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,        # 기본 연산
    tf.lite.OpsSet.SELECT_TF_OPS           # Flex ops 허용 ← 이거 안 넣으면 SplitV 등 안 됨
]

# 변환 수행
tflite_model = converter.convert()

# 저장
with open("best_float32.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite 변환 완료!")
