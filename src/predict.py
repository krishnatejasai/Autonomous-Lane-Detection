import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from augment import get_eval_transform
from dataset import LaneSegmentationDataset
from models import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"

DEFAULT_THRESHOLD = 0.5
DEFAULT_SAMPLES = 8

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
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


def denormalize_image(
    image_tensor: torch.Tensor,
) -> np.ndarray:
    image = (
        image_tensor.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    image = (
        image * IMAGENET_STD
        + IMAGENET_MEAN
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    overlay = image.copy()

    binary_mask = mask > 0

    color_array = np.array(
        color,
        dtype=np.float32,
    ) / 255.0

    overlay[binary_mask] = (
        (1.0 - alpha) * overlay[binary_mask]
        + alpha * color_array
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


def create_comparison_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    overlay = image.copy()

    ground_truth_mask = ground_truth > 0
    prediction_mask = prediction > 0

    correct_mask = (
        ground_truth_mask
        & prediction_mask
    )

    missed_mask = (
        ground_truth_mask
        & ~prediction_mask
    )

    false_positive_mask = (
        prediction_mask
        & ~ground_truth_mask
    )

    correct_color = (
        np.array(
            [255, 255, 0],
            dtype=np.float32,
        )
        / 255.0
    )

    missed_color = (
        np.array(
            [0, 255, 0],
            dtype=np.float32,
        )
        / 255.0
    )

    false_positive_color = (
        np.array(
            [255, 0, 0],
            dtype=np.float32,
        )
        / 255.0
    )

    overlay[correct_mask] = (
        (1.0 - alpha) * overlay[correct_mask]
        + alpha * correct_color
    )

    overlay[missed_mask] = (
        (1.0 - alpha) * overlay[missed_mask]
        + alpha * missed_color
    )

    overlay[false_positive_mask] = (
        (1.0 - alpha)
        * overlay[false_positive_mask]
        + alpha * false_positive_color
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
    number_of_samples: int,
) -> List[dict]:
    predictions = []

    for batch in data_loader:
        images = batch["image"].to(device)
        masks = batch["mask"]

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        predicted_masks = (
            probabilities >= threshold
        ).float()

        for index in range(images.shape[0]):
            image = denormalize_image(
                images[index]
            )

            ground_truth = (
                masks[index, 0]
                .cpu()
                .numpy()
            )

            prediction = (
                predicted_masks[index, 0]
                .cpu()
                .numpy()
            )

            probability_map = (
                probabilities[index, 0]
                .cpu()
                .numpy()
            )

            predictions.append(
                {
                    "image": image,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "probability_map": probability_map,
                    "image_path": batch["image_path"][index],
                }
            )

            if len(predictions) >= number_of_samples:
                return predictions

    return predictions


def save_individual_prediction(
    sample: dict,
    output_path: Path,
) -> None:
    image = sample["image"]
    ground_truth = sample["ground_truth"]
    prediction = sample["prediction"]

    prediction_overlay = create_overlay(
        image=image,
        mask=prediction,
        color=(255, 0, 0),
    )

    comparison_overlay = create_comparison_overlay(
        image=image,
        ground_truth=ground_truth,
        prediction=prediction,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
    )

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Road Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(
        prediction_overlay
    )
    axes[1, 0].set_title(
        "Prediction Overlay"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        comparison_overlay
    )
    axes[1, 1].set_title(
        "Yellow=Correct | Green=Missed | Red=False Positive"
    )
    axes[1, 1].axis("off")

    figure.suptitle(
        Path(sample["image_path"]).name,
        fontsize=12,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_prediction_grid(
    samples: List[dict],
    output_path: Path,
) -> None:
    rows = len(samples)

    figure, axes = plt.subplots(
        rows,
        4,
        figsize=(16, 4 * rows),
    )

    if rows == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    for row, sample in enumerate(samples):
        image = sample["image"]
        ground_truth = sample["ground_truth"]
        prediction = sample["prediction"]

        prediction_overlay = create_overlay(
            image=image,
            mask=prediction,
            color=(255, 0, 0),
        )

        comparison_overlay = create_comparison_overlay(
            image=image,
            ground_truth=ground_truth,
            prediction=prediction,
        )

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Road Image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(
            ground_truth,
            cmap="gray",
        )
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(
            prediction_overlay
        )
        axes[row, 2].set_title(
            "Prediction Overlay"
        )
        axes[row, 2].axis("off")

        axes[row, 3].imshow(
            comparison_overlay
        )
        axes[row, 3].set_title(
            "Yellow=Correct | Green=Missed | Red=FP"
        )
        axes[row, 3].axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative lane-segmentation "
            "predictions"
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
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
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
            f"Checkpoint not found: {checkpoint_path}"
        )

    model, checkpoint = load_model(
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

    dataset = LaneSegmentationDataset(
        split_file=(
            f"data/splits/{args.split}.json"
        ),
        image_height=image_height,
        image_width=image_width,
        transform=get_eval_transform(),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    samples = generate_predictions(
        model=model,
        data_loader=data_loader,
        device=device,
        threshold=args.threshold,
        number_of_samples=args.samples,
    )

    if not samples:
        raise RuntimeError(
            "No prediction samples were generated."
        )

    output_directory = (
        PREDICTION_DIR
        / args.model
        / args.split
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        output_path = (
            output_directory
            / f"prediction_{index:02d}.png"
        )

        save_individual_prediction(
            sample=sample,
            output_path=output_path,
        )

    grid_output = (
        PREDICTION_DIR
        / f"{args.model}_{args.split}_prediction_grid.png"
    )

    save_prediction_grid(
        samples=samples,
        output_path=grid_output,
    )

    print(
        "\n========== PREDICTION VISUALIZATION =========="
    )

    print(f"Model:         {args.model}")
    print(f"Split:         {args.split}")
    print(f"Device:        {device}")
    print(f"Samples:       {len(samples)}")
    print(f"Threshold:     {args.threshold}")
    print(f"Output folder: {output_directory}")
    print(f"Grid output:   {grid_output}")


if __name__ == "__main__":
    main()