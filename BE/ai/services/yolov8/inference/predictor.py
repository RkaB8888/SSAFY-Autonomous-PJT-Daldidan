# services/yolov8/inference/predictor.py

from PIL import Image
import numpy as np

from services.yolov8 import config
from services.yolov8.inference.backend_tflite import TFLiteYoloV8Backend
from services.yolov8.utils.preprocessing import preprocess_image
from services.yolov8.utils.postprocessing import postprocess_bbox, postprocess_seg


class YoloV8Predictor:
    def __init__(self, model_type: str = "bbox"):
        """
        Args:
            model_type: "bbox" or "seg"
        """
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == "bbox":
            model_path = config.MODEL_DIR / config.MODEL_FILES["bbox_int8"]
        elif self.model_type == "seg":
            model_path = config.MODEL_DIR / config.MODEL_FILES["seg_float32"]
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        print(f"[predictor] Loading model: {model_path}")
        return TFLiteYoloV8Backend(str(model_path))

    def predict(self, image: Image.Image | np.ndarray):
        # 1. Preprocess
        input_tensor, scale, pad = preprocess_image(
            image, target_size=config.INPUT_SIZE, dtype=self.model.input_dtype
        )

        # 2. Inference
        raw_output = self.model.infer(input_tensor)

        # 3. Postprocess
        if self.model_type == "bbox":
            return postprocess_bbox(
                raw_output[0],
                scale=scale,
                pad=pad,
                original_shape=image.size,
                conf_thres=config.CONF_THRES,
                iou_thres=config.IOU_THRES,
                target_class_id=47,  # apple
            )
        else:  # seg
            return postprocess_seg(
                raw_output,
                scale=scale,
                pad=pad,
                original_shape=image.size,
                conf_thres=config.CONF_THRES,
                iou_thres=config.IOU_THRES,
                target_class_id=47,
            )
