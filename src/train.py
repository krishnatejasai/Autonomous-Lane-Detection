import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from augment import get_eval_transform, get_train_transform
from dataset import LaneSegmentationDataset
from models import build_model
from utils import BCEDiceLoss


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 5
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def calculate_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    epsilon: float = 1e-7,
) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    true_positive = (
        predictions * targets
    ).sum()

    false_positive = (
        predictions * (1.0 - targets)
    ).sum()

    false_negative = (
        (1.0 - predictions) * targets
    ).sum()

    true_negative = (
        (1.0 - predictions)
        * (1.0 - targets)
    ).sum()

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
            + epsilon
        )
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
            + epsilon
        )
    )

    dice = (
        2.0 * true_positive
        / (
            2.0 * true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    iou = (
        true_positive
        / (
            true_positive
            + false_positive
            + false_negative
            + epsilon
        )
    )

    accuracy = (
        true_positive
        + true_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "dice": dice.item(),
        "iou": iou.item(),
        "accuracy": accuracy.item(),
    }


def create_data_loaders(
    batch_size: int,
    image_height: int,
    image_width: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = LaneSegmentationDataset(
        split_file="data/splits/train.json",
        image_height=image_height,
        image_width=image_width,
        transform=get_train_transform(),
    )

    validation_dataset = LaneSegmentationDataset(
        split_file="data/splits/val.json",
        image_height=image_height,
        image_width=image_width,
        transform=get_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


def run_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: AdamW = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "accuracy": 0.0,
    }

    progress_bar = tqdm(
        data_loader,
        desc="Training" if is_training else "Validating",
        leave=False,
    )

    for batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                loss.backward()
                optimizer.step()

        metrics = calculate_batch_metrics(
            logits=logits.detach(),
            targets=masks,
            threshold=threshold,
        )

        totals["loss"] += loss.item()

        for metric_name in metrics:
            totals[metric_name] += metrics[metric_name]

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{metrics['dice']:.4f}",
            iou=f"{metrics['iou']:.4f}",
        )

    number_of_batches = len(data_loader)

    return {
        metric_name: metric_value / number_of_batches
        for metric_name, metric_value in totals.items()
    }


def save_history(
    history: list,
    model_name: str,
) -> None:
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        METRICS_DIR
        / f"{model_name}_training_history.json"
    )

    csv_path = (
        METRICS_DIR
        / f"{model_name}_training_history.csv"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    if history:
        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history[0].keys(),
            )

            writer.writeheader()
            writer.writerows(history)


def train_model(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    image_height: int,
    image_width: int,
    threshold: float,
) -> None:
    set_seed()

    device = get_device()

    print(
        "\n========== AUTONOMOUS LANE "
        "DETECTION TRAINING =========="
    )

    print(f"Model:          {model_name}")
    print(f"Device:         {device}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(
        f"Image size:     "
        f"{image_height} x {image_width}"
    )

    train_loader, validation_loader = (
        create_data_loaders(
            batch_size=batch_size,
            image_height=image_height,
            image_width=image_width,
        )
    )

    print(
        f"Training samples:   "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_loader.dataset)}"
    )

    model = build_model(model_name).to(device)

    criterion = BCEDiceLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_name}_best.pt"
    )

    last_checkpoint_path = (
        CHECKPOINT_DIR
        / f"{model_name}_last.pt"
    )

    best_validation_dice = -1.0
    epochs_without_improvement = 0
    history = []

    training_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            threshold=threshold,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            threshold=threshold,
        )

        scheduler.step()

        epoch_duration = (
            time.perf_counter()
            - epoch_start_time
        )

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"],
            "val_dice": validation_metrics["dice"],
            "val_iou": validation_metrics["iou"],
            "val_precision": validation_metrics["precision"],
            "val_recall": validation_metrics["recall"],
            "val_accuracy": validation_metrics["accuracy"],
            "epoch_seconds": epoch_duration,
        }

        history.append(record)
        save_history(history, model_name)

        print(
            f"Train | "
            f"Loss={train_metrics['loss']:.4f} | "
            f"Dice={train_metrics['dice']:.4f} | "
            f"IoU={train_metrics['iou']:.4f}"
        )

        print(
            f"Valid | "
            f"Loss={validation_metrics['loss']:.4f} | "
            f"Dice={validation_metrics['dice']:.4f} | "
            f"IoU={validation_metrics['iou']:.4f} | "
            f"Precision="
            f"{validation_metrics['precision']:.4f} | "
            f"Recall="
            f"{validation_metrics['recall']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": validation_metrics,
            "image_height": image_height,
            "image_width": image_width,
            "threshold": threshold,
        }

        torch.save(
            checkpoint,
            last_checkpoint_path,
        )

        current_validation_dice = (
            validation_metrics["dice"]
        )

        if (
            current_validation_dice
            > best_validation_dice
        ):
            best_validation_dice = (
                current_validation_dice
            )

            epochs_without_improvement = 0

            torch.save(
                checkpoint,
                best_checkpoint_path,
            )

            print(
                "Saved new best checkpoint: "
                f"{best_checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                f"\nEarly stopping triggered "
                f"after {epoch} epochs."
            )
            break

    total_training_time = (
        time.perf_counter()
        - training_start_time
    )

    print(
        "\nTraining complete."
    )

    print(
        f"Best validation Dice: "
        f"{best_validation_dice:.4f}"
    )

    print(
        f"Training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN model for "
            "lane segmentation"
        )
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "unet"],
        default="baseline",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        image_height=args.image_height,
        image_width=args.image_width,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()