# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
RF-DETR 훈련 스크립트

사용법:
    python train.py --dataset_dir /path/to/dataset --epochs 100

지원 데이터셋 형식:
    - COCO (roboflow 내보내기): train/_annotations.coco.json
    - YOLO: data.yaml + train/valid/test 폴더
"""

import argparse
import os

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR 훈련")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="데이터셋 루트 경로 (COCO 또는 YOLO 형식)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=list(MODEL_MAP.keys()),
        help="모델 크기 선택 (기본값: base)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="총 훈련 에폭 수 (기본값: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="배치 크기 (기본값: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="학습률 (기본값: 1e-4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="체크포인트 및 결과 저장 경로 (기본값: output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cpu", "cuda", "mps", "auto"],
        help=f"훈련 디바이스 (기본값: {DEVICE})",
    )
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="roboflow",
        choices=["roboflow", "coco", "yolo"],
        help="데이터셋 형식 (기본값: roboflow)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="이어서 훈련할 체크포인트 경로",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="그래디언트 누적 스텝 수 (기본값: 4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="데이터로더 워커 수 (기본값: 2)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Weights & Biases 로깅 활성화",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="TensorBoard 로깅 활성화 (기본값: True)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="조기 종료 활성화",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="WandB 프로젝트 이름",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="WandB 런 이름",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="훈련에 사용할 데이터 분할 (기본값: train, val로 설정하면 검증 데이터로 훈련 가능)",
    )
    # Keypoint arguments
    parser.add_argument(
        "--keypoint_head",
        action="store_true",
        help="COCO keypoint 훈련 활성화 (annotations에 keypoints 필드 필요)",
    )
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=17,
        help="keypoint 개수 (COCO 기본값: 17, 기본값: 17)",
    )
    parser.add_argument(
        "--keypoint_loss_coef",
        type=float,
        default=5.0,
        help="keypoint 회귀 손실 가중치 (기본값: 5.0)",
    )
    parser.add_argument(
        "--set_cost_keypoint",
        type=float,
        default=5.0,
        help="헝가리안 매칭의 keypoint 비용 가중치 (기본값: 5.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"모델: {args.model}")
    logger.info(f"데이터셋: {args.dataset_dir}")
    logger.info(f"디바이스: {args.device}")
    logger.info(f"에폭: {args.epochs}, 배치 크기: {args.batch_size}")
    if args.keypoint_head:
        logger.info(f"Keypoint 훈련 활성화: num_keypoints={args.num_keypoints}")

    # output_dir에 checkpoint.pth가 있으면 자동으로 이어서 훈련
    if args.resume is None:
        auto_resume_path = os.path.join(args.output_dir, "checkpoint.pth")
        if os.path.isfile(auto_resume_path):
            logger.info(f"이전 체크포인트 발견, 자동 이어서 훈련: {auto_resume_path}")
            args.resume = auto_resume_path

    model_cls = MODEL_MAP[args.model]
    model = model_cls(keypoint_head=args.keypoint_head, num_keypoints=args.num_keypoints)

    os.makedirs(args.output_dir, exist_ok=True)

    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        dataset_file=args.dataset_format,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        resume=args.resume,
        tensorboard=args.tensorboard,
        wandb=args.wandb,
        early_stopping=args.early_stopping,
        project=args.project,
        run=args.run,
        train_split=args.train_split,
    )
    if args.keypoint_head:
        train_kwargs["keypoint_head"] = True
        train_kwargs["num_keypoints"] = args.num_keypoints
        train_kwargs["keypoint_loss_coef"] = args.keypoint_loss_coef
        train_kwargs["set_cost_keypoint"] = args.set_cost_keypoint

    model.train(**train_kwargs)

    logger.info(f"훈련 완료. 결과 저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
