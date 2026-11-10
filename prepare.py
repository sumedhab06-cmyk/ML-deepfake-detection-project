import argparse
from pathlib import Path

from deepfake.data import build_manifest


def main():
    parser = argparse.ArgumentParser(description="Extract the four archives and make leakage-aware splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    build_manifest(Path(__file__).resolve().parent, args.seed, args.val_fraction)


if __name__ == "__main__":
    main()
