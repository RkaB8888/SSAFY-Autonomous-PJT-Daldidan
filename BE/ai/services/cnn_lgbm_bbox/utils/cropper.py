# services/cnn_lgbm_bbox/utils/cropper.py
from PIL import Image, ImageDraw
import numpy as np


def crop_apple(img: Image.Image, ann: dict, use_seg: bool, resize: tuple):
    if use_seg:
        seg = ann.get("segmentation", [])
        if not seg:
            return None

        # flat list 또는 nested list 중 coords 가져오기
        coords = seg[0] if isinstance(seg[0], list) else seg

        # 최소 3개 포인트(6개 값)가 있는지 확인
        if len(coords) < 6 or len(coords) % 2 != 0:
            return None

        try:
            poly = np.array(coords, dtype=float).reshape(-1, 2)
        except Exception:
            # reshape 실패 시 skip
            return None

        # 정상적인 다각형인지 추가 검사 (예: 볼록성 등)
        # 여기서는 단순히 넉넉히 3개 이상만 확인
        if poly.shape[0] < 3:
            return None

        # 마스크 생성 & 크롭
        try:
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon([tuple(pt) for pt in poly], outline=1, fill=1)
            arr = np.array(img)
            arr[np.array(mask) == 0] = 0
            x0, y0 = poly.min(axis=0).astype(int)
            x1, y1 = poly.max(axis=0).astype(int)
            # bbox가 뒤집히거나 영역이 없으면 skip
            if x1 <= x0 or y1 <= y0:
                return None
            cropped = Image.fromarray(arr).crop((x0, y0, x1, y1))
        except Exception:
            return None
    else:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            return None
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            return None

        # 경계 밖 clamp
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, img.width), min(y + h, img.height)
        if x1 <= x0 or y1 <= y0:
            return None
        cropped = img.crop((x0, y0, x1, y1))

    return cropped.resize(resize, Image.Resampling.LANCZOS)
