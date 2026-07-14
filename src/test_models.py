import torch

from models import build_model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_model(model_name: str) -> None:
    model = build_model(model_name)

    dummy_input = torch.randn(
        2,
        3,
        256,
        512,
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nModel: {model_name}")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(
        "Trainable parameters: "
        f"{count_parameters(model):,}"
    )


def main() -> None:
    test_model("baseline")
    test_model("unet")


if __name__ == "__main__":
    main()