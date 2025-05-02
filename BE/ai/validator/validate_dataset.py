# =============================================================
# 📌 사용 예시 (터미널 실행)
# =============================================================
# 전체 유효성 검사 (결과 CSV 저장):
# python validate_dataset.py --mode all --img_dir ./images --json_dir ./jsons --save_csv result.csv
#
# 단일 이미지 기준 시각화 검사:
# python validate_dataset.py --mode image --img_path ./images/sample.jpg --json_dir ./jsons
#
# 단일 JSON 기준 시각화 검사:
# python validate_dataset.py --mode json --json_path ./jsons/sample.json --img_dir ./images
#
# 랜덤 샘플 N장 시각적 수동 검사 (실패 파일 로그 저장):
# python validate_dataset.py --mode random --img_dir ./images --json_dir ./jsons --n 20 --fail_log failed.txt
# =============================================================

import os
import json
import cv2
import numpy as np
import random
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from typing import List, Tuple, Dict, Any, Optional  # 타입 힌트를 위한 import

# 시각화에 사용할 색상 상수 정의
BBOX_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR for OpenCV)
SEG_COLOR: Tuple[int, int, int] = (255, 0, 0)  # Blue (BGR for OpenCV)


def load_image(img_path: Path) -> Optional[np.ndarray]:
    """
    이미지 파일을 불러옵니다. OpenCV로 시도하고 실패 시 Pillow로 폴백합니다.
    Pillow 로드 시 EXIF 회전 정보를 적용합니다.

    Args:
        img_path: 이미지 파일 경로 (Path 객체).

    Returns:
        불러온 이미지의 NumPy 배열 (BGR 채널) 또는 로딩 실패 시 None.
    """
    try:
        # OpenCV로 먼저 로드 시도
        image = cv2.imread(str(img_path))
        if image is not None:
            return image

        # OpenCV 실패 시 Pillow로 로드 시도
        pil_image = Image.open(img_path)
        # EXIF Orientation 정보 적용하여 이미지 회전/반전 (필요시)
        # 이 함수는 이미지를 변형하고 EXIF orientation 태그를 제거합니다.
        pil_image = ImageOps.exif_transpose(pil_image)

        # RGB로 변환 (알파 채널 등 제거)
        pil_image = pil_image.convert("RGB")

        image = np.array(pil_image)
        # Pillow는 RGB 순서이므로 OpenCV의 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    except Exception as e:
        print(f"[ERROR] 이미지 로딩 실패: {img_path} - {e}")
        return None


def validate_bbox(bbox: List[float], width: int, height: int) -> bool:
    """
    바운딩 박스 좌표가 이미지 경계 내에 있고 유효한지 검증합니다.

    Args:
        bbox: [x, y, w, h] 형태의 바운딩 박스 리스트.
        width: 이미지 너비.
        height: 이미지 높이.

    Returns:
        바운딩 박스가 유효하면 True, 그렇지 않으면 False.
    """
    if len(bbox) != 4:
        return False
    x, y, w, h = bbox
    return (
        0 <= x < width
        and 0 <= y < height
        and x + w <= width  # x + w는 너비와 같거나 작아야 함
        and y + h <= height  # y + h는 높이와 같거나 작아야 함
        and w > 0  # 너비는 0보다 커야 함
        and h > 0  # 높이는 0보다 커야 함
    )


def validate_segmentation(seg: List[float], width: int, height: int) -> bool:
    """
    세그멘테이션 좌표가 이미지 경계 내에 있고 유효한지 검증합니다.

    Args:
        seg: [x1, y1, x2, y2, ...] 형태의 세그멘테이션 좌표 리스트 (플랫 리스트).
        width: 이미지 너비.
        height: 이미지 높이.

    Returns:
        세그멘테이션이 유효하면 True, 그렇지 않으면 False.
    """
    # 최소 3개의 점 (6개 좌표) 필요
    if len(seg) < 6 or len(seg) % 2 != 0:
        return False
    # 모든 점이 이미지 경계 내에 있는지 확인
    for px, py in zip(seg[::2], seg[1::2]):
        if not (0 <= px < width and 0 <= py < height):
            return False
    return True


def visualize(
    image: np.ndarray,
    bbox: List[float],
    seg_points: List[Tuple[float, float]],
    title: Optional[str] = None,
):
    """
    이미지에 바운딩 박스와 세그멘테이션을 그려서 시각화합니다.

    Args:
        image: 불러온 이미지 NumPy 배열 (BGR).
        bbox: [x, y, w, h] 형태의 바운딩 박스 리스트.
        seg_points: [(x1, y1), (x2, y2), ...] 형태의 세그멘테이션 점 리스트.
        title: 플롯 제목 (선택 사항).
    """
    # OpenCV BGR 이미지를 Matplotlib RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_img = image_rgb.copy()

    # 바운딩 박스 그리기
    x, y, w, h = bbox
    # cv2.rectangle 함수는 int 좌표를 받음
    cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), BBOX_COLOR, 2)

    # 세그멘테이션 폴리곤 그리기
    if seg_points:
        # cv2.polylines는 [np.array([[[x1, y1]], [[x2, y2]], ...], dtype=int32)] 형태를 선호
        # 또는 단일 윤곽선인 경우 [np.array([[x1, y1], [x2, y2], ...], dtype=int32)] 형태도 가능
        # 여기서는 후자를 사용합니다.
        pts = [np.array(seg_points, dtype=np.int32)]
        cv2.polylines(vis_img, pts, isClosed=True, color=SEG_COLOR, thickness=2)

    # Matplotlib으로 이미지 표시
    plt.imshow(vis_img)
    plt.axis("off")  # 축 제거
    if title:
        plt.title(title)
    plt.show()


def check_pair(
    img_path: Path,
    json_path: Path,
    validate_only: bool = False,
    visualize_only: bool = False,
) -> Tuple[bool, bool]:
    """
    단일 이미지-JSON 쌍을 검사하고 유효성을 검증하며, 필요에 따라 시각화합니다.

    Args:
        img_path: 이미지 파일 경로 (Path 객체).
        json_path: JSON 어노테이션 파일 경로 (Path 객체).
        validate_only: 유효성 검사만 수행하고 시각화하지 않음 (기본값: False).
        visualize_only: 유효성 검사 결과를 반환하지 않고 시각화만 수행 (기본값: False).

    Returns:
        (바운딩 박스 유효성, 세그멘테이션 유효성) 튜플. 오류 발생 시 (False, False).
    """
    bbox_valid = False
    seg_valid = False

    try:
        # JSON 파일 로드 및 어노테이션 추출
        with open(json_path, "r", encoding="utf-8") as f:  # 인코딩 명시
            data: Dict[str, Any] = json.load(f)

        # 필요한 키가 있는지 확인
        if "annotations" not in data:
            print(f"[ERROR] {json_path.stem} : 'annotations' 키가 없습니다.")
            return False, False
        annotations = data["annotations"]

        if "bbox" not in annotations:
            print(
                f"[ERROR] {json_path.stem} : 'annotations' 내에 'bbox' 키가 없습니다."
            )
            return False, False
        bbox = annotations["bbox"]

        if "segmentation" not in annotations:
            print(
                f"[ERROR] {json_path.stem} : 'annotations' 내에 'segmentation' 키가 없습니다."
            )
            return False, False
        seg = annotations["segmentation"]

        # 세그멘테이션 좌표를 점 목록 [(x, y), ...] 형태로 변환
        # 세그멘테이션 데이터가 비어있을 수 있으므로 확인
        seg_points: List[Tuple[float, float]] = []
        if seg and isinstance(seg, list) and len(seg) % 2 == 0:
            seg_points = list(zip(seg[::2], seg[1::2]))
        elif seg:  # seg가 있지만 리스트가 아니거나 길이가 홀수인 경우
            print(
                f"[WARNING] {json_path.stem} : 유효하지 않은 세그멘테이션 데이터 형식입니다."
            )

        # 이미지 파일 로드
        image = load_image(img_path)
        if image is None:
            # load_image 함수에서 이미 오류 메시지 출력됨
            return False, False

        # 이미지 크기 가져오기
        h, w = image.shape[:2]

        # 유효성 검사 수행
        bbox_valid = validate_bbox(bbox, w, h)
        seg_valid = validate_segmentation(seg, w, h)

        # 시각화 (visualize_only 모드이고 validate_only 모드가 아닐 때)
        if visualize_only and not validate_only:
            if not bbox_valid:
                print(
                    f"[WARNING] {img_path.stem}: 바운딩 박스가 유효하지 않지만 시각화합니다."
                )
            if not seg_valid:
                print(
                    f"[WARNING] {img_path.stem}: 세그멘테이션이 유효하지 않지만 시각화합니다."
                )

            visualize(image, bbox, seg_points, title=img_path.name)

        # 유효성 검사 결과 반환 (validate_only 모드이거나 visualize_only 모드가 아닐 때)
        if validate_only or not visualize_only:
            return bbox_valid, seg_valid
        else:
            # visualize_only 모드일 때는 유효성 결과가 중요하지 않으므로 기본값 반환
            # (하지만 실제로는 validate_only=False && visualize_only=True 일 때만 이 분기에 오므로,
            #  시각화만 하고 True, True를 반환하여 오류로 기록되지 않게 하는 것이 목적일 수 있습니다.
            #  원 코드의 동작을 유지하기 위해 (False, False) 대신 (True, True)를 반환하여 오류 목록에 포함되지 않게 합니다.)
            return True, True  # 시각화 성공 가정

    except json.JSONDecodeError:
        print(f"[ERROR] {json_path.stem} : JSON 파싱 오류가 발생했습니다.")
        return False, False
    except KeyError as e:
        # 특정 키 누락은 위에서 처리하지만, 혹시 모를 다른 KeyError
        print(
            f"[ERROR] {json_path.stem} : JSON 데이터에서 예상치 못한 키 오류가 발생했습니다: {e}"
        )
        return False, False
    except Exception as e:
        # 그 외 예상치 못한 오류
        print(f"[ERROR] {img_path.stem} 검사 중 예상치 못한 오류 발생: {e}")
        return False, False


def check_all(img_dir: str, json_dir: str, save_csv: Optional[str] = None):
    """
    지정된 폴더 및 하위 폴더 내의 모든 이미지-JSON 쌍을 검사합니다.

    Args:
        img_dir: 이미지가 있는 폴더 경로.
        json_dir: JSON 어노테이션이 있는 폴더 경로.
        save_csv: 오류 목록을 저장할 CSV 파일 경로 (선택 사항).
    """
    # rglob를 사용하여 재귀적으로 파일 검색
    img_files: Dict[str, Path] = {f.stem: f for f in Path(img_dir).rglob("*.jpg")}
    json_files: Dict[str, Path] = {f.stem: f for f in Path(json_dir).rglob("*.json")}

    # 이미지 또는 JSON 파일이 있는 모든 고유한 파일 이름(stem) 가져오기
    all_keys = sorted(set(img_files.keys()) | set(json_files.keys()))
    errors: List[List[str]] = []

    print(f"[INFO] 총 {len(all_keys)}개의 파일 쌍(후보)을 검사합니다.")

    for idx, key in enumerate(all_keys, 1):
        print(f"[진행] {idx}/{len(all_keys)}: {key}")
        img_path = img_files.get(key)
        json_path = json_files.get(key)

        # 파일 누락 확인
        if not img_path:
            print(f"[WARNING] {key}: 이미지 파일(.jpg)이 없습니다.")
            errors.append([key, "missing_image"])
            continue
        if not json_path:
            print(f"[WARNING] {key}: JSON 파일(.json)이 없습니다.")
            errors.append([key, "missing_json"])
            continue

        # 쌍 검사 및 유효성 검증만 수행
        vbox, vseg = check_pair(img_path, json_path, validate_only=True)

        # 유효성 검사 실패 시 오류 기록
        if not vbox or not vseg:
            err_types: List[str] = []
            if not vbox:
                err_types.append("invalid_bbox")
            if not vseg:
                err_types.append("invalid_seg")
            errors.append([key, ",".join(err_types)])
            print(f"[WARNING] {key}: 유효성 검사 실패 - {', '.join(err_types)}")

    print(f"[DONE] 총 검사 파일 쌍 수: {len(all_keys)} / 오류 수: {len(errors)}")

    # 오류 목록 CSV 저장
    if save_csv:
        try:
            with open(save_csv, "w", newline="", encoding="utf-8") as f:  # 인코딩 명시
                writer = csv.writer(f)
                writer.writerow(["filename", "error_type"])
                writer.writerows(errors)
            print(f"[INFO] 오류 결과 저장 완료: {save_csv}")
        except Exception as e:
            print(f"[ERROR] 오류 결과를 CSV에 저장하는데 실패했습니다: {e}")


def check_random(
    img_dir: str, json_dir: str, n: int = 10, fail_log: Optional[str] = None
):
    """
    이미지-JSON 쌍 중에서 n개를 랜덤으로 선택하여 수동으로 시각적 검사를 수행합니다.

    Args:
        img_dir: 이미지가 있는 폴더 경로.
        json_dir: JSON 어노테이션이 있는 폴더 경로.
        n: 랜덤으로 선택할 파일 쌍의 수 (기본값: 10).
        fail_log: 수동 검사에서 'n'(아니오)이라고 응답한 파일 이름을 저장할 파일 경로 (선택 사항).
    """
    img_files: Dict[str, Path] = {f.stem: f for f in Path(img_dir).rglob("*.jpg")}
    json_files: Dict[str, Path] = {f.stem: f for f in Path(json_dir).rglob("*.json")}

    # 이미지와 JSON이 모두 존재하는 파일 이름(stem) 목록
    matched_keys: List[str] = [k for k in img_files.keys() if k in json_files]

    if not matched_keys:
        print("[INFO] 이미지와 JSON 파일 쌍이 발견되지 않았습니다.")
        return

    # 랜덤 샘플 선택
    n_samples = min(n, len(matched_keys))
    sample_keys = random.sample(matched_keys, n_samples)

    print(
        f"[INFO] 총 {len(matched_keys)}개의 쌍 중 {n_samples}개를 랜덤 선택하여 수동 검사를 시작합니다."
    )
    fails: List[str] = []

    for i, name in enumerate(sample_keys):
        img_path = img_files[name]
        json_path = json_files[name]

        print(f"\n--- 검사 {i+1}/{n_samples}: {name} ---")

        # 쌍 시각화 (유효성 검사는 check_pair 내부에서 수행되지만, visualize_only 모드이므로 결과는 사용 안 함)
        # check_pair 함수는 visualize_only 모드일 때 True, True를 반환하도록 수정했습니다.
        _, _ = check_pair(img_path, json_path, visualize_only=True)

        # 사용자 피드백 받기
        while True:
            res = (
                input(
                    f"{name} → 어노테이션이 올바른가요? (y: 예 / n: 아니오 / exit: 중단): "
                )
                .strip()
                .lower()
            )
            if res == "y":
                break
            elif res == "n":
                fails.append(name)
                print(f"[INFO] {name} 파일은 실패 목록에 추가되었습니다.")
                break
            elif res == "exit":
                print("[INFO] 수동 확인이 중단되었습니다.")
                break
            else:
                print("잘못된 입력입니다. 'y', 'n', 또는 'exit'를 입력하세요.")

        if res == "exit":
            break  # 수동 검사 루프 중단

    # 수동 실패 목록 저장
    if fail_log and fails:
        try:
            existing_fails = set()
            fail_path = Path(fail_log)

            # 이미 존재하는 경우 기존 실패 목록 읽기
            if fail_path.exists():
                with open(fail_path, "r", encoding="utf-8") as f:
                    existing_fails = set(line.strip() for line in f if line.strip())

            # 새로운 실패 항목과 기존 항목을 합쳐 중복 제거
            combined_fails = existing_fails.union(set(fails))

            with open(fail_log, "w", encoding="utf-8") as f:  # 인코딩 명시
                for name in sorted(combined_fails):
                    f.write(name + "\n")
            print(
                f"\n[INFO] 수동 검사 실패 파일 목록 저장 완료: {fail_log} ({len(combined_fails)}개)"
            )
        except Exception as e:
            print(f"[ERROR] 수동 실패 목록을 파일에 저장하는데 실패했습니다: {e}")
    elif fail_log and not fails:
        print(f"\n[INFO] 수동 검사에서 실패한 파일이 없습니다.")


def check_single_image(img_path_str: str, json_dir: str):
    """
    단일 이미지 파일에 대해 대응하는 JSON 파일을 찾아 검사하고 시각화합니다.

    Args:
        img_path_str: 검사할 이미지 파일 경로 문자열.
        json_dir: JSON 어노테이션이 있는 폴더 경로.
    """
    img_path = Path(img_path_str)
    if not img_path.exists():
        print(f"[ERROR] 지정된 이미지 파일이 존재하지 않습니다: {img_path}")
        return

    stem = img_path.stem
    # json_dir 내에서 동일한 이름을 가진 JSON 파일 검색
    jsons_found: List[Path] = list(Path(json_dir).rglob(f"{stem}.json"))

    if not jsons_found:
        print(
            f"[ERROR] {stem}.json 파일을 '{json_dir}' 또는 하위 폴더에서 찾을 수 없습니다."
        )
        return
    elif len(jsons_found) > 1:
        print(
            f"[WARNING] {stem}.json 파일이 여러 개 발견되었습니다. 첫 번째 파일({jsons_found[0]})을 사용합니다."
        )
        for j_path in jsons_found:
            print(f"- 발견된 JSON: {j_path}")

    json_path = jsons_found[0]  # 첫 번째 발견된 JSON 파일 사용

    print(f"[INFO] 이미지: {img_path} 에 대응하는 JSON: {json_path} 검사를 시작합니다.")
    # 단일 검사는 시각화 목적
    _, _ = check_pair(img_path, json_path, visualize_only=True)


def check_single_json(json_path_str: str, img_dir: str):
    """
    단일 JSON 파일에 대해 대응하는 이미지 파일을 찾아 검사하고 시각화합니다.

    Args:
        json_path_str: 검사할 JSON 파일 경로 문자열.
        img_dir: 이미지가 있는 폴더 경로.
    """
    json_path = Path(json_path_str)
    if not json_path.exists():
        print(f"[ERROR] 지정된 JSON 파일이 존재하지 않습니다: {json_path}")
        return

    stem = json_path.stem
    # img_dir 내에서 동일한 이름을 가진 jpg 파일 검색
    imgs_found: List[Path] = list(
        Path(img_dir).rglob(f"{stem}.jpg")
    )  # 원본과 동일하게 jpg만 검색

    if not imgs_found:
        print(
            f"[ERROR] {stem}.jpg 파일을 '{img_dir}' 또는 하위 폴더에서 찾을 수 없습니다."
        )
        return
    elif len(imgs_found) > 1:
        print(
            f"[WARNING] {stem}.jpg 파일이 여러 개 발견되었습니다. 첫 번째 파일({imgs_found[0]})을 사용합니다."
        )
        for i_path in imgs_found:
            print(f"- 발견된 이미지: {i_path}")

    img_path = imgs_found[0]  # 첫 번째 발견된 이미지 파일 사용

    print(f"[INFO] JSON: {json_path} 에 대응하는 이미지: {img_path} 검사를 시작합니다.")
    # 단일 검사는 시각화 목적
    _, _ = check_pair(img_path, json_path, visualize_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="이미지 어노테이션(바운딩 박스/세그멘테이션) 검사 및 시각화 스크립트"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "random", "image", "json"],
        required=True,
        help="실행 모드: 'all' (전체 검사), 'random' (랜덤 샘플 수동 검사), 'image' (단일 이미지 검사), 'json' (단일 JSON 검사)",
    )
    # 단일 파일 모드를 위해 --img_path 와 --json_path 유지
    parser.add_argument(
        "--img_path", type=str, help="단일 이미지 파일 경로 (mode='image' 사용)"
    )
    parser.add_argument(
        "--json_path", type=str, help="단일 JSON 파일 경로 (mode='json' 사용)"
    )
    # 폴더 모드를 위해 --img_dir 와 --json_dir 유지
    parser.add_argument(
        "--img_dir",
        type=str,
        help="이미지 파일이 있는 루트 폴더 경로 (mode='all' 또는 'random' 사용)",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        help="JSON 파일이 있는 루트 폴더 경로 (mode='all' 또는 'random' 또는 'image' 사용)",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="random 모드에서 검사할 샘플 수 (기본값: 10)"
    )
    parser.add_argument(
        "--save_csv", type=str, help="all 모드에서 오류 결과를 저장할 CSV 파일 경로"
    )
    parser.add_argument(
        "--fail_log",
        type=str,
        help="random 모드에서 수동 검사 실패 목록을 저장할 파일 경로",
    )

    args = parser.parse_args()

    # 인자 유효성 검사 (필요한 인자가 제공되었는지 확인)
    if args.mode in ["all", "random"]:
        if not args.img_dir or not args.json_dir:
            parser.error(
                f"mode '{args.mode}' 실행 시 --img_dir 와 --json_dir 는 필수입니다."
            )
    elif args.mode == "image":
        if not args.img_path or not args.json_dir:
            parser.error(
                f"mode '{args.mode}' 실행 시 --img_path 와 --json_dir 는 필수입니다."
            )
    elif args.mode == "json":
        if not args.json_path or not args.img_dir:
            parser.error(
                f"mode '{args.mode}' 실행 시 --json_path 와 --img_dir 는 필수입니다."
            )

    # 모드에 따른 기능 실행
    if args.mode == "all":
        check_all(args.img_dir, args.json_dir, save_csv=args.save_csv)
    elif args.mode == "random":
        check_random(args.img_dir, args.json_dir, n=args.n, fail_log=args.fail_log)
    elif args.mode == "image":
        check_single_image(args.img_path, args.json_dir)
    elif args.mode == "json":
        check_single_json(args.json_path, args.img_dir)
