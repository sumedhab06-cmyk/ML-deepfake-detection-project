import csv
import random
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ARCHIVES = {
    "train_real": ("celebdf (1).tar.gz", "celebdf"),
    "train_fake": ("celebdf_recons (1).tar.gz", "celebdf_recons"),
    "test_real": ("cacd_auth.tar.gz", "cacd_auth"),
    "test_fake": ("cacd_deepfake.tar.gz", "cacd_deepfake"),
}
FIELDS = ("split", "label", "identity", "group", "path")


def extract_archives(root):
    root = Path(root).resolve()
    archive_dir = root / "data" / "archives"
    extract_dir = root / "data" / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    for _, (filename, folder) in ARCHIVES.items():
        archive = archive_dir / filename
        if not archive.is_file():
            raise FileNotFoundError(
                f"Missing {archive}. Keep the original archive name."
            )

        marker = extract_dir / f".{folder}.complete"
        if marker.exists():
            continue

        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()

            # Validate every archive path before extracting any member.
            for member in members:
                target = (extract_dir / member.name).resolve()
                if not target.is_relative_to(extract_dir):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsupported archive member: {member.name}")
                if not member.name.split("/")[0] == folder:
                    raise ValueError(f"Unexpected archive directory: {member.name}")

            for member in members:
                if member.isdir():
                    (extract_dir / member.name).mkdir(parents=True, exist_ok=True)
                else:
                    destination = extract_dir / member.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source_file = handle.extractfile(member)
                    if source_file is None:
                        raise OSError(f"Could not read archive member: {member.name}")
                    with source_file, destination.open("wb") as output:
                        shutil.copyfileobj(source_file, output)

        marker.touch()

    return extract_dir


def build_manifest(root, seed=42, val_fraction=0.2):
    """Create train/val/test manifest with source-video groups kept together.

    Labels: 0 = real, 1 = fake. Validation groups are keyed by both identity
    and group to avoid collisions when two identities have similarly named videos.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    root = Path(root).resolve()
    extract_dir = extract_archives(root)
    groups = defaultdict(list)
    rows = []

    for kind, (_, folder) in ARCHIVES.items():
        label = int(kind.endswith("fake"))
        image_paths = sorted((extract_dir / folder).glob("*/*.jpg"))

        for path in image_paths:
            identity = path.parent.name
            if kind.startswith("train"):
                match = re.fullmatch(r"(id\d+_\d+)_frame\d+face\d+\.jpg", path.name)
                if match is None:
                    raise ValueError(f"Unexpected training filename: {path}")
                group = match.group(1)
                groups[identity].append(group)
            else:
                group = path.stem

            rows.append({
                "split": "test" if kind.startswith("test") else "train",
                "label": label,
                "identity": identity,
                "group": group,
                "path": path.relative_to(root).as_posix(),
            })

    required = ((split, label) for split in ("train", "test") for label in (0, 1))
    if not rows or any(
        not any(row["split"] == split and row["label"] == label for row in rows)
        for split, label in required
    ):
        raise ValueError("Missing images in at least one dataset split/class")

    rng = random.Random(seed)
    validation_groups = set()
    for identity, identity_groups in sorted(groups.items()):
        unique_groups = sorted(set(identity_groups))
        if len(unique_groups) < 2:
            raise ValueError(f"Need at least two source videos for identity {identity}")
        rng.shuffle(unique_groups)
        count = max(1, round(len(unique_groups) * val_fraction))
        for group in unique_groups[:count]:
            validation_groups.add((identity, group))

    for row in rows:
        if row["split"] == "train" and (row["identity"], row["group"]) in validation_groups:
            row["split"] = "val"

    # Ensure each original/reconstruction frame pair stays in the same split
    # and both labels are present for each pair.
    paired = defaultdict(set)
    pair_splits = defaultdict(set)
    for row in rows:
        if row["split"] in ("train", "val"):
            key = (row["identity"], row["group"], Path(row["path"]).name)
            paired[key].add(row["label"])
            pair_splits[key].add(row["split"])

    if any(labels != {0, 1} for labels in paired.values()):
        raise ValueError("Training original/reconstruction pairs do not match")
    if any(len(splits) != 1 for splits in pair_splits.values()):
        raise ValueError("An original/reconstruction pair crosses train and validation")

    manifest = root / "data" / "splits.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("Created", manifest)
    for split in ("train", "val", "test"):
        counts = Counter(row["label"] for row in rows if row["split"] == split)
        print(split, dict(counts), "total:", sum(counts.values()))
    return manifest


def load_rows(root, split):
    path = Path(root) / "data" / "splits.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 'python prepare.py' first: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["split"] == split]


def image_transform(training=False):
    from torchvision import transforms

    operations = [transforms.Resize((224, 224))]
    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(operations)


class FaceDataset:
    def __init__(self, root, split, training=False):
        self.root = Path(root).resolve()
        self.rows = load_rows(self.root, split)
        self.transform = image_transform(training)
        if not self.rows:
            raise ValueError(f"No images in split {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.root / row["path"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"])
