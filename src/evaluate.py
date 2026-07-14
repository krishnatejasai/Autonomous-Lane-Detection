import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from augment import get_eval_transform
from dataset import LaneSegmentationDataset
from models import build_model
from utils import BCEDiceLoss, SegmentationMetricTracker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

DEFAULT_BATCH_SIZE = 4
DEFAULT_THRESHOLD = 0.5


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = build_model(model_name)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    metric_tracker = SegmentationMetricTracker(
        threshold=threshold
    )

    total_loss = 0.0
    total_images = 0
    total_inference_time = 0.0

    progress_bar = tqdm(
        data_loader,
        desc="Evaluating",
    )

    for batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if device.type == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()

        logits = model(images)

        if device.type == "mps":
            torch.mps.synchronize()

        inference_time = (
            time.perf_counter()
            - start_time
        )

        loss = criterion(logits, masks)

        metric_tracker.update(
            logits=logits,
            targets=masks,
        )

        batch_size = images.shape[0]

        total_loss += (
            loss.item() * batch_size
        )

        total_images += batch_size
        total_inference_time += inference_time

    metrics = metric_tracker.compute()

    average_loss = (
        total_loss / total_images
    )

    latency_ms = (
        total_inference_time
        / total_images
        * 1000.0
    )

    fps = (
        total_images
        / total_inference_time
        if total_inference_time > 0
        else 0.0
    )

    metrics.update(
        {
            "loss": average_loss,
            "images_evaluated": total_images,
            "average_latency_ms": latency_ms,
            "fps": fps,
        }
    )

    return metrics


def print_metrics(
    model_name: str,
    split_name: str,
    checkpoint_path: Path,
    metrics: Dict[str, float],
) -> None:
    print(
        "\n========== LANE SEGMENTATION "
        "EVALUATION =========="
    )

    print(f"Model:             {model_name}")
    print(f"Split:             {split_name}")
    print(f"Checkpoint:        {checkpoint_path}")
    print(
        f"Images evaluated:  "
        f"{metrics['images_evaluated']}"
    )

    print("\nSegmentation Metrics")

    print(
        f"Pixel Accuracy:    "
        f"{metrics['pixel_accuracy']:.4f}"
    )

    print(
        f"Precision:         "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:            "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Dice / F1:         "
        f"{metrics['dice']:.4f}"
    )

    print(
        f"IoU:               "
        f"{metrics['iou']:.4f}"
    )

    print(
        f"Specificity:       "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Loss:              "
        f"{metrics['loss']:.4f}"
    )

    print("\nRuntime")

    print(
        f"Latency:           "
        f"{metrics['average_latency_ms']:.2f} ms/image"
    )

    print(
        f"Throughput:        "
        f"{metrics['fps']:.2f} FPS"
    )

    print("\nConfusion Counts")

    print(
        f"True Positive:     "
        f"{metrics['true_positive']}"
    )

    print(
        f"True Negative:     "
        f"{metrics['true_negative']}"
    )

    print(
        f"False Positive:    "
        f"{metrics['false_positive']}"
    )

    print(
        f"False Negative:    "
        f"{metrics['false_negative']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained lane "
            "segmentation model"
        )
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "unet"],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    args = parser.parse_args()

    device = get_device()

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (
            CHECKPOINT_DIR
            / f"{args.model}_best.pt"
        )
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = (
            PROJECT_ROOT / checkpoint_path
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    model, checkpoint = load_checkpoint(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    image_height = checkpoint.get(
        "image_height",
        256,
    )

    image_width = checkpoint.get(
        "image_width",
        512,
    )

    split_file = (
        f"data/splits/{args.split}.json"
    )

    dataset = LaneSegmentationDataset(
        split_file=split_file,
        image_height=image_height,
        image_width=image_width,
        transform=get_eval_transform(),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = BCEDiceLoss()

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        threshold=args.threshold,
    )

    output = {
        "model": args.model,
        "split": args.split,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "threshold": args.threshold,
        "image_height": image_height,
        "image_width": image_width,
        "metrics": metrics,
    }

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        METRICS_DIR
        / f"{args.model}_{args.split}_metrics.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print_metrics(
        model_name=args.model,
        split_name=args.split,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )

    print(
        f"\nMetrics saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()