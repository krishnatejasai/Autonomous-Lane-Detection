import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (
            probabilities * targets
        ).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum()
            + targets.sum()
            + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.bce(logits, targets)
            + self.dice(logits, targets)
        )


class SegmentationMetricTracker:
    def __init__(
        self,
        threshold: float = 0.5,
        epsilon: float = 1e-7,
    ) -> None:
        self.threshold = threshold
        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.true_positive = 0.0
        self.true_negative = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= self.threshold
        ).float()

        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)

        self.true_positive += float(
            (predictions * targets).sum().item()
        )

        self.true_negative += float(
            (
                (1.0 - predictions)
                * (1.0 - targets)
            ).sum().item()
        )

        self.false_positive += float(
            (
                predictions
                * (1.0 - targets)
            ).sum().item()
        )

        self.false_negative += float(
            (
                (1.0 - predictions)
                * targets
            ).sum().item()
        )

    def compute(self) -> dict:
        tp = self.true_positive
        tn = self.true_negative
        fp = self.false_positive
        fn = self.false_negative
        eps = self.epsilon

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        dice = (
            2.0 * tp
            / (
                2.0 * tp
                + fp
                + fn
                + eps
            )
        )

        iou = (
            tp
            / (
                tp
                + fp
                + fn
                + eps
            )
        )

        pixel_accuracy = (
            tp + tn
        ) / (
            tp
            + tn
            + fp
            + fn
            + eps
        )

        specificity = (
            tn
            / (
                tn
                + fp
                + eps
            )
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": dice,
            "dice": dice,
            "iou": iou,
            "pixel_accuracy": pixel_accuracy,
            "specificity": specificity,
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
        }