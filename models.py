import torch.nn as nn
from torchvision import models


NAMES = ("mobilenet_v3", "efficientnet_b0", "resnet50")


def make_model(name, pretrained=True):
    """Create a two-class classifier: class 0 = real, class 1 = fake."""
    if name == "mobilenet_v3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError(f"Unknown model {name}; choose from {NAMES}")
    return model
