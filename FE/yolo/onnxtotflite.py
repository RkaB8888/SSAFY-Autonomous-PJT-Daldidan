from onnx2tf import convert

convert(
    input_onnx_file_path="best.onnx",
    output_folder_path="./tflite_model",
    output_tflite=True
)
