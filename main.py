import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_name: str, extra_args: list[str]) -> None:
    script_path = PROJECT_ROOT / "src" / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        *extra_args,
    ]

    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Command-line interface for the "
            "Autonomous Driving Lane Detection project."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train a lane-segmentation model.",
    )
    train_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model.",
    )
    evaluate_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate prediction visualizations.",
    )
    predict_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
    )

    figures_parser = subparsers.add_parser(
        "figures",
        help="Generate evaluation figures.",
    )
    figures_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect the TuSimple dataset.",
    )
    inspect_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    command_to_script = {
        "train": "train.py",
        "evaluate": "evaluate.py",
        "predict": "predict.py",
        "figures": "generate_figures.py",
        "inspect": "inspect_dataset.py",
    }

    run_script(
        script_name=command_to_script[args.command],
        extra_args=args.args,
    )


if __name__ == "__main__":
    main()