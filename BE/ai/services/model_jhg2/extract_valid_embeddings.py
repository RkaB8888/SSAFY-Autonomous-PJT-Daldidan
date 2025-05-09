# ai/services/model_jhg2/extract_valid_embeddings.py
import json, pathlib, glob
import numpy as np, tqdm
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader

from services.model_jhg2.config import CACHE_DIR, VALID_IMAGES_DIR, VALID_JSONS_DIR
from services.model_jhg2.utils.cnn_feature_extractor import extract_batch


class CroppedDataset(Dataset):
    def __init__(
        self, img_dir: pathlib.Path, json_dir: pathlib.Path, resize=(256, 256)
    ):
        # 1) 원래 있던 (이미지, json) 쌍을 모두 수집
        raw_pairs = sorted(
            [
                (p, json_dir / f"{p.stem}.json")
                for p in img_dir.glob("*.jpg")
                if (json_dir / f"{p.stem}.json").exists()
            ]
        )

        # 2) sugar_content가 반드시 있는 것만 필터링
        self.pairs = []
        for img_p, js_p in raw_pairs:
            data = json.loads(js_p.read_text(encoding="utf-8"))
            coll = data.get("collection", {})
            if (
                coll.get("sugar_content") is not None
                or coll.get("sugar_content_nir") is not None
            ):
                ## sugar_content가 존재해야만 추가
                # if data.get("collection", {}).get("sugar_content") is not None:
                self.pairs.append((img_p, js_p))
        self.resize = resize

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, js_p = self.pairs[idx]
        try:
            # — 이미지 열어서 crop & resize
            img = Image.open(img_p).convert("RGB")
            with open(js_p, "r", encoding="utf-8") as f:
                data = json.load(f)

            # # 1) COCO 폴리곤 불러오기
            # seg = data["annotations"]["segmentation"]
            # # flat list인지 nested list인지 검사
            # if isinstance(seg[0], list):
            #     poly = np.array(seg[0]).reshape(-1, 2)
            # else:
            #     poly = np.array(seg).reshape(-1, 2)

            # # 2) 마스크 생성
            # mask = Image.new("L", img.size, 0)
            # ImageDraw.Draw(mask).polygon(
            #     [tuple(point) for point in poly], outline=1, fill=1
            # )
            # mask_arr = np.array(mask)

            # # 3) 이미지에 마스크 적용 (배경을 검정으로)
            # img_arr = np.array(img)
            # img_arr[mask_arr == 0] = 0

            # # 4) 마스크 영역의 bounding box로 크롭
            # x0, y0 = poly.min(axis=0).astype(int)
            # x1, y1 = poly.max(axis=0).astype(int)
            # crop = Image.fromarray(img_arr).crop((x0, y0, x1, y1))
            # crop = crop.resize(self.resize, Image.Resampling.LANCZOS)

            x, y, w, h = map(int, data["annotations"]["bbox"])
            crop = img.crop((x, y, x + w, y + h)).resize(
                self.resize, Image.Resampling.LANCZOS
            )
            arr = np.array(crop, copy=False)

            # — 레이블(당도)
            coll = data.get("collection", {})
            # sugar = coll.get("sugar_content")
            sugar = coll.get("sugar_content") or coll.get("sugar_content_nir")

            if sugar is None:
                print(f"[Warning] sugar_content 누락, 건너뜁니다: {js_p.name}")
                return None  # 해당 샘플을 건너뜀

            sugar = float(sugar)

            return arr, sugar, img_p.stem

        except Exception as e:
            print(f"[Warning] 에러 발생 (idx={idx}): {e}")
            return None  # 에러 시 None 반환


def build_and_cache_embeddings(
    img_dir: pathlib.Path = pathlib.Path(VALID_IMAGES_DIR),
    json_dir: pathlib.Path = pathlib.Path(VALID_JSONS_DIR),
):
    """
    • VALID_IMAGES_DIR/VALID_JSONS_DIR 를 순회하며 crop→임베딩 추출→메모리맵+npy로 저장
    """
    ds = CroppedDataset(img_dir, json_dir)
    dl = DataLoader(
        ds,
        batch_size=1024,
        num_workers=32,
        collate_fn=lambda x: [s for s in x if s is not None],
        prefetch_factor=4,
        pin_memory=False,
        persistent_workers=False,
    )

    N = len(ds)
    D = 1280
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    feat_path = CACHE_DIR / "valid_embeddings.npy"
    label_path = CACHE_DIR / "valid_labels.npy"
    stems_path = CACHE_DIR / "valid_stems.npy"

    feats = np.memmap(feat_path, dtype=np.float32, mode="w+", shape=(N, D))
    labels = np.memmap(label_path, dtype=np.float32, mode="w+", shape=(N,))
    stems = []

    idx = 0
    for batch in tqdm.tqdm(dl, total=len(dl), ncols=80):
        # 빈 배치는 건너뛰기
        if not batch:
            continue
        imgs, sugars, batch_stems = zip(*batch)
        batch_np = np.stack(imgs, axis=0)
        vecs = extract_batch(batch_np)
        B = vecs.shape[0]

        feats[idx : idx + B] = vecs
        labels[idx : idx + B] = sugars
        stems.extend(batch_stems)
        idx += B

    feats.flush()
    labels.flush()
    np.save(stems_path, np.array(stems))
    print(
        f"[Saved] valid embeddings→{feat_path}, labels→{label_path}, stems→{stems_path}"
    )


if __name__ == "__main__":
    build_and_cache_embeddings()
