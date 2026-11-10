
import csv
from collections import Counter, defaultdict

counts = defaultdict(Counter)

with open("data/splits.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        counts[row["split"]][row["label"]] += 1

for split in ("train", "val", "test"):
    print(f"{split}: {dict(counts[split])}")
    print(f"  Total: {sum(counts[split].values())}")

print("Total images:", sum(sum(c.values()) for c in counts.values()))
