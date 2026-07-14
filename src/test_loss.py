import torch

from utils import BCEDiceLoss


criterion = BCEDiceLoss()

pred = torch.randn(
    2,
    1,
    256,
    512,
)

target = torch.randint(
    0,
    2,
    (
        2,
        1,
        256,
        512,
    ),
).float()

loss = criterion(pred, target)

print(loss)