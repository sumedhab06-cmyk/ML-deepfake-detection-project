import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .data import FaceDataset
from .models import make_model


def choose_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    return torch.device(requested)


@torch.inference_mode()
def predict_loader(model, loader, device):
    """Return labels, fake-class probabilities, and total inference time."""
    model.eval()
    labels, probabilities = [], []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for images, batch_labels in loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        logits = model(images)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                f"Expected model output shape [batch, 2], got {tuple(logits.shape)}"
            )
        # Model class index 0 = real; class index 1 = fake.
        fake_probabilities = torch.softmax(logits, dim=1)[:, 1]
        probabilities.extend(fake_probabilities.cpu().tolist())
        labels.extend(batch_labels.tolist())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float), seconds


def find_best_threshold(labels, probabilities):
    """Choose the validation threshold that maximizes fake-class F1."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    return float(thresholds[int(np.argmax(f1_values))])


def metrics(labels, probabilities, seconds=None, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    result = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predicted),
        "balanced_accuracy": balanced_accuracy_score(labels, predicted),
        "precision_fake": precision_score(labels, predicted, zero_division=0),
        "recall_fake": recall_score(labels, predicted, zero_division=0),
        "specificity": specificity,
        "f1_fake": f1_score(labels, predicted, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else None,
        "confusion_matrix_real_fake": cm.tolist(),
        "samples": int(len(labels)),
    }
    if seconds is not None:
        result["seconds_per_image_batch_amortized"] = seconds / max(len(labels), 1)
        result["total_inference_seconds"] = seconds
    return {
        key: float(value) if isinstance(value, (np.floating, np.integer)) else value
        for key, value in result.items()
    }


def save_plots(labels, probabilities, directory, threshold=0.5):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    predicted = np.asarray(probabilities) >= threshold
    cm = confusion_matrix(labels, predicted, labels=[0, 1])

    ConfusionMatrixDisplay(
        cm, display_labels=["Real", "Fake"]
    ).plot(cmap="Blues", colorbar=False)
    plt.title(f"Confusion matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close()

    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc_value = roc_auc_score(labels, probabilities)
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(directory / "roc_curve.png", dpi=160)
        plt.close()


def evaluate_checkpoint(root, checkpoint_path, batch_size=32, workers=0, device_name="auto"):
    root = Path(root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    name = checkpoint["model_name"]
    model = make_model(name, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = FaceDataset(root, "test")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    labels, probabilities, seconds = predict_loader(model, loader, device)
    threshold = float(checkpoint.get("threshold", 0.5))
    report = metrics(labels, probabilities, seconds, threshold=threshold)
    report.update(
        model=name,
        checkpoint=str(checkpoint_path),
        split="test",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        note=(
            "Cross-source CACD test: training uses Celeb-DF originals/reconstructions; "
            "test uses CACD authentic/deepfake images. Treat results as dataset-specific; "
            "identity-disjoint generalization has not been established."
        ),
    )

    output = root / "outputs" / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    save_plots(labels, probabilities, output, threshold=threshold)
    print(json.dumps(report, indent=2))
    return report
