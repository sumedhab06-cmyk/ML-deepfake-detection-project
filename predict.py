import argparse
from pathlib import Path

import torch
from PIL import Image

from deepfake.data import image_transform
from deepfake.evaluation import choose_device
from deepfake.models import make_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict one already-cropped facial image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = make_model(saved["model_name"], pretrained=False)
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    with Image.open(args.image) as image:
        tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = torch.softmax(model(tensor), dim=1)[0, 1].item()
    print(f"Prediction: {'Fake' if probability >= 0.5 else 'Real'}; P(Fake)={probability:.4f}")
    print("For face crops similar to this dataset; a probability is not a guarantee of authenticity.")
