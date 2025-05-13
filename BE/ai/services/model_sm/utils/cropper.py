# services/model_sm/utils/cropper.py
from PIL import Image, ImageDraw
import numpy as np


def crop_apple(img: Image.Image, ann: dict, use_seg: bool) -> Image.Image | None:
    if use_seg:
        seg = ann.get("segmentation", [])
        if not seg:
            return None

        coords = seg[0] if isinstance(seg[0], list) else seg
        if len(coords) < 6 or len(coords) % 2 != 0:
            return None

        try:
            poly = np.array(coords, dtype=float).reshape(-1, 2)
        except Exception:
            return None

        if poly.shape[0] < 3:
            return None

        try:
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon([tuple(pt) for pt in poly], outline=1, fill=1)
            arr = np.array(img)
            arr[np.array(mask) == 0] = 0
            x0, y0 = poly.min(axis=0).astype(int)
            x1, y1 = poly.max(axis=0).astype(int)
            if x1 <= x0 or y1 <= y0:
                return None
            return Image.fromarray(arr).crop((x0, y0, x1, y1))
        except Exception:
            return None

    else:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            return None
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            return None
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, img.width), min(y + h, img.height)
        if x1 <= x0 or y1 <= y0:
            return None
        return img.crop((x0, y0, x1, y1))
