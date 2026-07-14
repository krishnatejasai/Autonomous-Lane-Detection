from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tusimple"
    / "tusimple_preprocessed"
    / "training"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "dataset_sample.png"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_image_folder() -> Path:
    candidate_names = {
        "images",
        "image",
        "frames",
        "clips",
        "rgb",
    }

    for folder in DATA_ROOT.iterdir():
        if (
            folder.is_dir()
            and folder.name.lower() in candidate_names
        ):
            return folder

    folders = [
        folder
        for folder in DATA_ROOT.iterdir()
        if folder.is_dir()
        and "mask" not in folder.name.lower()
    ]

    if not folders:
        raise FileNotFoundError(
            f"No source-image folder found inside {DATA_ROOT}"
        )

    return folders[0]


def find_mask_folder() -> Path:
    folders = [
        folder
        for folder in DATA_ROOT.iterdir()
        if folder.is_dir()
        and "mask" in folder.name.lower()
    ]

    if not folders:
        raise FileNotFoundError(
            f"No mask folder found inside {DATA_ROOT}"
        )

    return folders[0]


def collect_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    image_folder = find_image_folder()
    mask_folder = find_mask_folder()

    images = collect_files(image_folder)
    masks = collect_files(mask_folder)

    if not images:
        raise FileNotFoundError(
            f"No images found in {image_folder}"
        )

    if not masks:
        raise FileNotFoundError(
            f"No masks found in {mask_folder}"
        )

    image_map = {
        path.stem: path
        for path in images
    }

    mask_map = {
        path.stem: path
        for path in masks
    }

    shared_names = sorted(
        set(image_map) & set(mask_map)
    )

    if not shared_names:
        raise RuntimeError(
            "No matching image-mask filenames were found."
        )

    sample_name = shared_names[0]

    image = np.asarray(
        Image.open(image_map[sample_name]).convert("RGB")
    )

    mask = np.asarray(
        Image.open(mask_map[sample_name]).convert("L")
    )

    binary_mask = mask > 0

    overlay = image.copy()
    overlay[binary_mask] = (
        0.45 * overlay[binary_mask]
        + 0.55 * np.array([255, 0, 0])
    ).astype(np.uint8)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Road Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Lane Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Lane Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_FILE,
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()

    print("Dataset inspection complete.")
    print(f"Image folder: {image_folder}")
    print(f"Mask folder:  {mask_folder}")
    print(f"Image files:  {len(images)}")
    print(f"Mask files:   {len(masks)}")
    print(f"Matched pairs: {len(shared_names)}")
    print(f"Sample output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()