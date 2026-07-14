import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curves(model_name: str):
    history_file = METRICS_DIR / f"{model_name}_training_history.csv"

    if not history_file.exists():
        print(f"Missing {history_file}")
        return

    history = pd.read_csv(history_file)

    epochs = history["epoch"]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{model_name}_loss_curve.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"], label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"{model_name.upper()} Dice Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / f"{model_name}_dice_curve.png",
        dpi=300,
    )

    plt.close()


def compare_models():

    baseline = json.load(
        open(
            METRICS_DIR / "baseline_val_metrics.json"
        )
    )

    unet = json.load(
        open(
            METRICS_DIR / "unet_val_metrics.json"
        )
    )

    metrics = [
        "precision",
        "recall",
        "dice",
        "iou",
    ]

    baseline_values = [
        baseline["metrics"][m]
        for m in metrics
    ]

    unet_values = [
        unet["metrics"][m]
        for m in metrics
    ]

    import numpy as np

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8,5))

    plt.bar(
        x-width/2,
        baseline_values,
        width,
        label="Baseline",
    )

    plt.bar(
        x+width/2,
        unet_values,
        width,
        label="U-Net",
    )

    plt.xticks(
        x,
        [m.upper() for m in metrics],
    )

    plt.ylim(0,1)

    plt.ylabel("Score")

    plt.title("Baseline vs U-Net")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_comparison.png",
        dpi=300,
    )

    plt.close()


def main():

    plot_training_curves("baseline")

    plot_training_curves("unet")

    compare_models()

    print("\nFigures saved to:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()