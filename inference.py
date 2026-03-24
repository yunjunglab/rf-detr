# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
RF-DETR 추론 스크립트

사용법:
    # 단일 이미지
    python inference.py --checkpoint output/checkpoint_best_total.pth --image /path/to/image.jpg

    # 디렉토리 내 모든 이미지
    python inference.py --checkpoint output/checkpoint_best_total.pth --image /path/to/images/

    # keypoint 모델 (checkpoint에서 자동 감지)
    python inference.py --checkpoint output_kpt/checkpoint_best_total.pth --image /path/to/image.jpg
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import supervision as sv
from PIL import Image

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.config import DEVICE
from rfdetr.util.logger import get_logger

logger = get_logger()

MODEL_MAP = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge,
}

MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]

# COCO 17-keypoint skeleton connections (0-indexed)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

KEYPOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170),
]


def load_checkpoint_args(checkpoint_path: str) -> object:
    """체크포인트에서 저장된 훈련 args를 로드합니다."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("args", None)


def run_inference(
    model: "RFDETRBase",
    image_path: str,
    threshold: float,
) -> tuple[sv.Detections, np.ndarray | None, np.ndarray]:
    """이미지에 대해 추론을 실행하고 검출 결과와 keypoint를 반환합니다.

    Args:
        model: RFDETR 모델 인스턴스.
        image_path: 입력 이미지 파일 경로.
        threshold: 신뢰도 임계값 (이 값 이상인 검출만 반환).

    Returns:
        Tuple of (sv.Detections, keypoints array or None, original RGB image array).
        keypoints shape: (N, K, 3) — [x, y, visibility] in absolute pixel coords.
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_tensor = F.to_tensor(img)
    h, w = img_tensor.shape[1:]

    device = model.model.device
    img_tensor = img_tensor.to(device)
    img_tensor = F.normalize(img_tensor, MEANS, STDS)
    img_tensor = F.resize(img_tensor, (model.model.resolution, model.model.resolution))
    batch = img_tensor.unsqueeze(0)

    model.model.model.eval()
    with torch.no_grad():
        raw = model.model.model(batch)
        if isinstance(raw, tuple):
            pred_dict = {"pred_logits": raw[1], "pred_boxes": raw[0]}
            if len(raw) > 2:
                pred_dict["pred_keypoints"] = raw[2]
        else:
            pred_dict = raw

        target_sizes = torch.tensor([[h, w]], device=device)
        results = model.model.postprocess(pred_dict, target_sizes=target_sizes)

    result = results[0]
    keep = result["scores"] > threshold

    boxes = result["boxes"][keep].float().cpu().numpy()
    scores = result["scores"][keep].float().cpu().numpy()
    labels = result["labels"][keep].cpu().numpy()

    keypoints = None
    if "keypoints" in result:
        keypoints = result["keypoints"][keep].cpu().numpy()  # (N, K, 3)

    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels.astype(int),
    )
    return detections, keypoints, img_np


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    radius: int = 5,
    draw_skeleton: bool = True,
) -> np.ndarray:
    """이미지에 keypoint와 skeleton을 그립니다.

    Args:
        image: BGR 이미지 배열.
        keypoints: (N, K, 3) 배열 — [x, y, visibility] in absolute pixels.
        radius: keypoint 원의 반지름 (픽셀).
        draw_skeleton: COCO skeleton 연결선을 그릴지 여부.

    Returns:
        Annotated BGR image.
    """
    img = image.copy()
    num_kpts = keypoints.shape[1]

    for person_kpts in keypoints:
        if draw_skeleton and num_kpts == 17:
            for i, j in COCO_SKELETON:
                xi, yi, vi = person_kpts[i]
                xj, yj, vj = person_kpts[j]
                if vi > 0 and vj > 0:
                    cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), (200, 200, 200), 2)

        for k in range(num_kpts):
            x, y, v = person_kpts[k]
            if v > 0:
                color = KEYPOINT_COLORS[k % len(KEYPOINT_COLORS)]
                cv2.circle(img, (int(x), int(y)), radius, color, -1)
                cv2.circle(img, (int(x), int(y)), radius, (0, 0, 0), 1)

    return img


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="RF-DETR 추론")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="체크포인트 파일 경로 (예: output/checkpoint_best_total.pth)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="입력 이미지 경로 또는 이미지가 있는 디렉토리 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_inference",
        help="결과 이미지 저장 경로 (기본값: output_inference)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="신뢰도 임계값 (기본값: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=list(MODEL_MAP.keys()),
        help="모델 크기 (기본값: base). checkpoint의 args에서 자동 감지 불가 시 사용.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cpu", "cuda", "mps", "auto"],
        help=f"추론 디바이스 (기본값: {DEVICE})",
    )
    parser.add_argument(
        "--no_skeleton",
        action="store_true",
        help="keypoint skeleton 연결선을 그리지 않음",
    )
    return parser.parse_args()


def main() -> None:
    """추론 메인 함수."""
    args = parse_args()

    # 체크포인트 args에서 모델 설정 자동 감지
    logger.info(f"체크포인트 로드: {args.checkpoint}")
    ckpt_args = load_checkpoint_args(args.checkpoint)

    keypoint_head = False
    num_keypoints = 17
    if ckpt_args is not None:
        keypoint_head = getattr(ckpt_args, "keypoint_head", False)
        num_keypoints = getattr(ckpt_args, "num_keypoints", 17)

    logger.info(f"모델: {args.model}, keypoint_head={keypoint_head}, num_keypoints={num_keypoints}")

    model_cls = MODEL_MAP[args.model]
    model = model_cls(
        pretrain_weights=args.checkpoint,
        keypoint_head=keypoint_head,
        num_keypoints=num_keypoints,
        device=args.device,
    )

    os.makedirs(args.output, exist_ok=True)

    # 이미지 경로 수집
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if os.path.isdir(args.image):
        image_paths = [
            os.path.join(args.image, f)
            for f in sorted(os.listdir(args.image))
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        if not image_paths:
            logger.error(f"이미지를 찾을 수 없습니다: {args.image}")
            return
    else:
        image_paths = [args.image]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    class_names = model.class_names  # {id: name}

    for image_path in image_paths:
        logger.info(f"처리 중: {image_path}")

        detections, keypoints, img_rgb = run_inference(model, image_path, args.threshold)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 바운딩 박스 및 레이블 그리기
        det_labels = [
            f"{class_names.get(int(c) + 1, class_names.get(int(c), str(c)))} {s:.2f}"
            for c, s in zip(detections.class_id, detections.confidence)
        ]
        img_bgr = box_annotator.annotate(img_bgr, detections=detections)
        img_bgr = label_annotator.annotate(img_bgr, detections=detections, labels=det_labels)

        # keypoint 그리기
        if keypoints is not None and len(keypoints) > 0:
            img_bgr = draw_keypoints(img_bgr, keypoints, draw_skeleton=not args.no_skeleton)

        out_path = os.path.join(args.output, os.path.basename(image_path))
        cv2.imwrite(out_path, img_bgr)
        logger.info(f"저장 완료: {out_path} ({len(detections)} 검출)")

    logger.info(f"완료. 결과 저장 위치: {args.output}")


if __name__ == "__main__":
    main()
