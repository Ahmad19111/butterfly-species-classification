"""ResNet50 mit ImageNet-Gewichten und neuem Klassifikationskopf."""
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

HEAD_PREFIX = "fc."


def build_resnet50(n_classes, weights_name):
    weights = ResNet50_Weights[weights_name]
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model, weights


def set_backbone_trainable(model, trainable):
    for name, p in model.named_parameters():
        if not name.startswith(HEAD_PREFIX):
            p.requires_grad = trainable
