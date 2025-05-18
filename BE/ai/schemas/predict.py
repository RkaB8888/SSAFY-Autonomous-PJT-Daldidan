# ai/schemas/predict.py
from typing import List, Optional
from pydantic import BaseModel


class BBox(BaseModel):  # (x, y, w, h)  or  (xmin, ymin, xmax, ymax) 중 하나로 통일
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class Segmentation(BaseModel):  # COCO-style 폴리곤 점 목록
    points: List[
        List[List[float]]
    ]  # a list of contours, each contour is a list of [x, y] pairs


class ApplePred(BaseModel):
    id: int
    sugar_content: float
    bbox: Optional[BBox] = None
    segmentation: Optional[Segmentation] = None


class PredictResponse(BaseModel):
    results: List[ApplePred]
