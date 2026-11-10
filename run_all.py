import argparse
import subprocess
import sys
from pathlib import Path

from deepfake.models import NAMES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare, train, and test all three architectures")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "prepare.py"], cwd=root, check=True)
    for name in NAMES:
        subprocess.run([sys.executable, "train.py", "--model", name, "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size), "--device", args.device], cwd=root, check=True)
        subprocess.run([sys.executable, "evaluate.py", "--model", name, "--batch-size", str(args.batch_size),
                        "--device", args.device], cwd=root, check=True)
    print("All results are in outputs/<model>/test_metrics.json and plots.")
