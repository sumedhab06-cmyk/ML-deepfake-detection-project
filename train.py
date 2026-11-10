import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from deepfake.data import FaceDataset
from deepfake.evaluation import (
    choose_device,
    find_best_threshold,
    metrics,
    predict_loader,
)
from deepfake.models import NAMES, make_model


def train(args):
    root = Path(__file__).resolve().parent
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")

    train_data = FaceDataset(root, "train", training=True)
    val_data = FaceDataset(root, "val")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output = root / "outputs" / args.model
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)

        val_labels, val_probabilities, _ = predict_loader(model, val_loader, device)
        threshold = find_best_threshold(val_labels, val_probabilities)
        report = metrics(
            val_labels,
            val_probabilities,
            threshold=threshold,
        )
        val_auc = report["roc_auc"]
        if val_auc is None:
            raise RuntimeError("Validation set must contain both real and fake images")
        scheduler.step(val_auc)

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_data),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "threshold": threshold,
            **{
                key: report[key]
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "precision_fake",
                    "recall_fake",
                    "f1_fake",
                    "roc_auc",
                )
            },
        }
        history.append(record)

        with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(history)

        print(json.dumps(record), flush=True)

        # Keep the checkpoint with the best validation ROC-AUC.
        if val_auc > best_auc:
            best_auc = val_auc
            stale = 0
            torch.save(
                {
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_auc": best_auc,
                    "threshold": threshold,
                    "class_to_index": {"real": 0, "fake": 1},
                    "image_size": 224,
                },
                output / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    print(
        f"Best checkpoint: {output / 'best.pt'} | "
        f"validation AUC: {best_auc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a binary real/fake face-image classifier"
    )
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from random initialization instead of transfer learning",
    )
    train(parser.parse_args())
