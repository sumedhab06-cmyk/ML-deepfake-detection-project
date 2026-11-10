import argparse
from pathlib import Path

from deepfake.evaluation import evaluate_checkpoint
from deepfake.models import NAMES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the held-out CACD test split"
    )
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    checkpoint = root / "outputs" / args.model / "best.pt"
    evaluate_checkpoint(
        root,
        checkpoint,
        batch_size=args.batch_size,
        workers=args.workers,
        device_name=args.device,
    )
