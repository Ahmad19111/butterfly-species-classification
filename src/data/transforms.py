"""Transforms für Training und Auswertung erstellen."""
import torch
from timm.data import resolve_model_data_config
from torchvision.transforms import v2 as T


def norm_from_torchvision(weights):
    t = weights.transforms()
    return list(t.mean), list(t.std)


def norm_from_timm(model):
    c = resolve_model_data_config(model)
    return list(c["mean"]), list(c["std"])


def build_transforms(train, mean, std, cfg):
    if train:
        steps = [
            T.RandomResizedCrop(cfg["img_size"], scale=tuple(cfg["rrc_scale"])),
            T.RandomHorizontalFlip(cfg["hflip_p"]),
        ]
    else:
        steps = [T.Resize(cfg["resize_size"]), T.CenterCrop(cfg["img_size"])]
    return T.Compose(steps + [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])
