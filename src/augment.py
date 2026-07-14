import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512


def get_train_transform() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=15,
                p=0.3,
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.04),
                p=0.2,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def get_eval_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )