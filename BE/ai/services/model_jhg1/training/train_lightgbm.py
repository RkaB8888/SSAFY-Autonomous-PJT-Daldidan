import json
import joblib
from pathlib import Path
from typing import List, Tuple
import concurrent.futures  # 병렬 처리를 위한 라이브러리 임포트
import os  # CPU 코어 개수 확인을 위해 임포트

import lightgbm as lgb
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold

# common_utils에서 필요한 함수 임포트 (기존 코드와 동일)
from common_utils.image_cropper import crop_bbox_from_json

# services.model_jhg1.utils.feature_extractors에서 extract_features 임포트 (기존 코드와 동일)
from services.model_jhg1.utils.feature_extractors import extract_features
from services.model_jhg1.config import IMAGES_DIR, JSONS_DIR, MODEL_SAVE_PATH


# 단일 이미지 처리를 위한 헬퍼 함수 정의
# 이 함수가 병렬로 실행될 작업 단위가 됩니다.
def process_single_image(
    image_path: Path, jsons_dir: Path
) -> Tuple[np.ndarray, int, List[str]]:
    """단일 이미지 파일에서 피처와 당도 정보를 추출하는 헬퍼 함수"""
    json_path = jsons_dir / (image_path.stem + ".json")
    if not json_path.exists():
        # 당도 정보가 없거나 JSON 파일이 없는 경우, None을 반환하여 무시하도록 처리
        return None

    try:
        # 이미지 자르기
        crop_img, _ = crop_bbox_from_json(image_path, json_path)
        if crop_img is None:
            # tqdm.write(f"[무시] 손상된 이미지: {image_path.name}") # 병렬 처리 시 출력 순서 문제 발생 가능성 있음
            return None

        # 피처 추출 (이 부분은 여전히 CPU에서 실행됨)
        feats, names = extract_features(crop_img)

        # 당도 정보 로드
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        collection = data.get("collection", {})
        sugar = collection.get("sugar_content")
        if sugar is None:
            sugar = collection.get("sugar_content_nir")

        if sugar is None:
            # tqdm.write(f"[무시] 당도 정보 없음: {image_path.name}") # 병렬 처리 시 출력 순서 문제 발생 가능성 있음
            return None

        # 유효한 결과 반환: 피처 벡터, 당도, 피처 이름 (피처 이름은 한 번만 필요하지만, 편의상 같이 반환)
        return feats, sugar, names

    except Exception as e:
        # tqdm.write(f"[Error] {image_path.name}: {e}") # 병렬 처리 시 출력 순서 문제 발생 가능성 있음
        return None  # 오류 발생 시 None 반환


# load_dataset 함수를 병렬 처리로 수정
def load_dataset(images_dir: Path, jsons_dir: Path):
    X, y = [], []
    feature_names: List[str] = None  # 한번만 채워둘 리스트

    image_files = sorted(
        list(images_dir.glob("*.jpg"))
    )  # glob 결과를 리스트로 변환하여 길이를 얻습니다.
    total_files = len(image_files)
    print(f"총 이미지 파일 수: {total_files}")

    # 병렬 처리를 위한 Executor 설정
    # ProcessPoolExecutor는 CPU 코어를 활용하여 계산이 많은 작업에 적합합니다.
    # ThreadPoolExecutor는 I/O 대기(파일 읽기 등)가 많은 작업에 적합하지만, 계산에는 GIL 제약이 있습니다.
    # 여기서는 피처 추출 계산이 많으므로 ProcessPoolExecutor가 더 적합할 수 있습니다.
    # 사용할 CPU 코어 수를 설정합니다. 기본값은 시스템의 코어 수입니다.
    max_workers = os.cpu_count()  # 사용 가능한 모든 코어 사용
    # 또는 특정 개수로 설정: max_workers = 8

    # tqdm과 concurrent.futures를 함께 사용하여 진행률 표시
    # ProcessPoolExecutor 사용 시 feature_names를 안전하게 한 번만 설정하는 로직 필요
    # 결과를 비동기적으로 수집
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 각 파일에 대해 작업을 제출하고 Future 객체를 얻습니다.
        future_to_file = {
            executor.submit(process_single_image, image_path, jsons_dir): image_path
            for image_path in image_files
        }

        # tqdm.as_completed를 사용하여 완료되는 순서대로 결과를 처리합니다.
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=total_files,
            desc="Extracting features (Parallel)",
        ):
            result = future.result()  # 작업 결과 가져오기
            if result is not None:
                feats, sugar, names = result
                X.append(feats)
                y.append(sugar)
                # feature_names는 첫 번째 유효한 결과에서만 저장합니다.
                if feature_names is None:
                    feature_names = names
            # else: # None이 반환된 경우는 이미 함수 내부에서 무시하기로 결정된 경우 (JSON 없음, 이미지 손상, 당도 정보 없음 등)
            # 무시된 파일에 대한 개별 메시지 출력을 병렬 환경에서는 지양하는 것이 좋습니다.
            # 필요하다면 별도로 로깅 시스템을 구축해야 합니다.

    print(f"✅ 유효 샘플 수: {len(X)} / 전체 시도 파일: {total_files}")

    # numpy 배열로 변환
    if len(X) == 0:
        print("경고: 유효한 샘플이 하나도 없습니다.")
        return np.array([]), np.array([]), []

    return np.array(X), np.array(y), feature_names


# train_lightgbm 및 main 함수는 기존과 동일
def train_lightgbm(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], save_path: Path
):
    if len(X) == 0:
        print("학습할 데이터가 없습니다. train_lightgbm 함수를 건너뜁니다.")
        return

    # -- 상수 피처 제거 --
    selector = VarianceThreshold(threshold=0.0)
    X_reduced = selector.fit_transform(X)
    print(f"▶ 분산 임계값 적용 후 샘플: {X_reduced.shape}")

    # -- 학습 --
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        device="gpu",  # <== GPU 사용 설정 (LightGBM 학습 단계에서 사용)
        gpu_use_dp=True,  # (선택) 더 안정적인 처리
        random_state=42,
        # 추가적인 GPU 관련 설정이 필요할 수 있습니다.
        # 예를 들어, gpu_platform_id, gpu_device_id 등이 있지만,
        # CUDA_VISIBLE_DEVICES를 설정하면 일반적으로 LightGBM이 알아서 찾습니다.
    )
    print("▶ LightGBM 학습 시작...")
    model.fit(X_reduced, y)
    print("✅ LightGBM 학습 완료.")

    # -- 모델 + selector + 원본 feature_names 한꺼번에 덤프 --
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "selector": selector, "feature_names": feature_names},
        save_path,
    )
    print(f"✅ LightGBM + meta 저장 완료: {save_path}")


def main():
    # 데이터를 로딩하고 피처를 추출합니다. (이제 이 단계가 병렬 처리됩니다)
    X, y, feature_names = load_dataset(IMAGES_DIR, JSONS_DIR)

    if len(X) > 0:
        print(f"▶ 최종 샘플: {X.shape}, y 분포: {np.unique(y)}")
        # 데이터 로딩/피처 추출이 끝나면 LightGBM 학습을 시작합니다.
        # 이 단계에서 GPU를 사용하게 됩니다.
        train_lightgbm(X, y, feature_names, MODEL_SAVE_PATH)
    else:
        print("▶ 유효한 데이터가 없어 학습을 진행하지 않습니다.")


if __name__ == "__main__":
    main()
