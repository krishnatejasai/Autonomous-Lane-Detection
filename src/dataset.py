import json
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_IMAGE_WIDTH = 512


class LaneSegmentationDataset(Dataset):
    def __init__(
        self,
        split_file: Union[str, Path],
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        transform: Optional[Callable] = None,
    ) -> None:
        self.split_file = Path(split_file)

        if not self.split_file.is_absolute():
            self.split_file = PROJECT_ROOT / self.split_file

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self.split_file}"
            )

        with open(
            self.split_file,
            "r",
            encoding="utf-8",
        ) as file:
            self.samples = json.load(file)

        if not self.samples:
            raise ValueError(
                f"No samples found in {self.split_file}"
            )

        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Union[torch.Tensor, str]]:
        sample = self.samples[index]

        image_path = Path(sample["image"])
        mask_path = Path(sample["mask"])

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        if not mask_path.is_absolute():
            mask_path = PROJECT_ROOT / mask_path

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        if mask is None:
            raise FileNotFoundError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (
                self.image_width,
                self.image_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (
                self.image_width,
                self.image_height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }