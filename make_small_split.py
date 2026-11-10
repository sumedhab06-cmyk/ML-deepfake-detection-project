"""Sample paired frames within every train/validation video; keep test intact."""
import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = ("split", "label", "identity", "group", "path")


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected splits.csv columns")
        return list(reader)


def write_manifest(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def create_small_split(root, fraction=0.25, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be > 0 and <= 1")
    manifest = root / "data" / "splits.csv"
    backup = root / "data" / "splits_full.csv"
    if not manifest.is_file():
        raise FileNotFoundError("Run python prepare.py first")
    if not backup.exists():
        shutil.copy2(manifest, backup)
    rows = read_manifest(backup)  # Always sample from the full original manifest.
    by_video = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["split"] in ("train", "val"):
            key = (row["split"], row["identity"], row["group"])
            by_video[key][Path(row["path"]).name].append(row)
    selected_paths = set()
    rng = random.Random(seed)
    for video, frames in sorted(by_video.items()):
        if any({r["label"] for r in pair} != {"0", "1"} or len(pair) != 2
               for pair in frames.values()):
            raise ValueError(f"Unmatched real/fake frame in {video}")
        filenames = sorted(frames)
        chosen = rng.sample(filenames, max(1, round(len(filenames) * fraction)))
        for filename in chosen:
            selected_paths.update(r["path"] for r in frames[filename])
    subset = [r for r in rows if r["split"] == "test" or r["path"] in selected_paths]
    write_manifest(manifest, subset)
    print(f"Saved full manifest to {backup}")
    print(f"Using {fraction:.0%} of paired frames in each train/validation video")
    for split in ("train", "val", "test"):
        print(split, dict(Counter(r["label"] for r in subset if r["split"] == split)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a smaller paired training/validation split")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restore", action="store_true", help="Restore the full manifest")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.restore:
        backup = root / "data" / "splits_full.csv"
        if not backup.exists():
            parser.error("No full manifest backup found")
        shutil.copy2(backup, root / "data" / "splits.csv")
        print("Restored full manifest")
    else:
        create_small_split(root, args.fraction, args.seed)
