from torch.utils.data import DataLoader

from augment import (
    get_eval_transform,
    get_train_transform,
)
from dataset import LaneSegmentationDataset


def main() -> None:
    train_dataset = LaneSegmentationDataset(
        split_file="data/splits/train.json",
        transform=get_train_transform(),
    )

    val_dataset = LaneSegmentationDataset(
        split_file="data/splits/val.json",
        transform=get_eval_transform(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    sample = train_dataset[0]

    print("\nSingle sample")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape:  {sample['mask'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Mask dtype:  {sample['mask'].dtype}")
    print(f"Mask min:    {sample['mask'].min().item()}")
    print(f"Mask max:    {sample['mask'].max().item()}")

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nBatch")
    print(f"Images shape: {batch['image'].shape}")
    print(f"Masks shape:  {batch['mask'].shape}")


if __name__ == "__main__":
    main()