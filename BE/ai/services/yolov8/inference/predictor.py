# services/yolov8/inference/predictor.py

from PIL import Image
import numpy as np

from services.yolov8 import config
from services.yolov8.inference.backend_tflite import TFLiteYoloV8Backend
from services.yolov8.utils.preprocessing import preprocess_image
from services.yolov8.utils.postprocessing import postprocess_coco, postprocess_custom


class YoloV8Predictor:
    def __init__(self, model_type: str = "coco"):
        """
        Args:
            model_type: "coco" or "custom"
        """
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        return TFLiteYoloV8Backend(config.DEFAULT_MODEL_PATH)

    def predict(self, image: Image.Image | np.ndarray):
        # 1. Preprocess
        input_tensor, scale, pad = preprocess_image(
            image, target_size=config.INPUT_SIZE, dtype=self.model.input_dtype
        )

        # 2. Inference
        raw_output = self.model.infer(input_tensor)

        # 3. Postprocess
        if self.model_type == "coco":
            return postprocess_coco(
                raw_output,
                scale=scale,
                pad=pad,
                original_shape=image.size,
                conf_thres=config.CONF_THRES,
                iou_thres=config.IOU_THRES,
                target_class_id=47,  # apple
            )
        else:
            return postprocess_custom(
                raw_output,
                scale=scale,
                pad=pad,
                original_shape=image.size,
                conf_thres=config.CONF_THRES,
                iou_thres=config.IOU_THRES,
            )
