"""Train- und Val/Test-Ansichten der Bilder anzeigen."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.models import ResNet50_Weights

from src.data.dataset import ButterflyDataset
from src.data.transforms import build_transforms, norm_from_torchvision
from src.utils.repro import set_seed

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "metadata" / "augmentations_train.jpg"
OUT_DPI = 150
JPEG_QUALITY = 90


def to_img(x, mean, std):
    x = x * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn_resnet50.yaml")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--n-aug", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    weights = ResNet50_Weights[cfg["model"]["weights"]]
    mean, std = norm_from_torchvision(weights)
    train_tf = build_transforms(True, mean, std, cfg["transforms"])
    eval_tf = build_transforms(False, mean, std, cfg["transforms"])

    ds = ButterflyDataset("train")
    idx = torch.randperm(len(ds))[:args.n_images].tolist()
    n_cols = args.n_aug + 1
    fig, axes = plt.subplots(args.n_images, n_cols, figsize=(2.2 * n_cols, 2.3 * args.n_images))
    for r, i in enumerate(idx):
        img, label = ds[i]
        views = [eval_tf(img)] + [train_tf(img) for _ in range(args.n_aug)]
        for c, v in enumerate(views):
            ax = axes[r, c]
            ax.imshow(to_img(v, mean, std))
            ax.axis("off")
            if r == 0:
                ax.set_title("val/test" if c == 0 else f"train {c}", fontsize=9)
        axes[r, 0].text(-0.05, 0.5, ds.class_names[label], transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=OUT_DPI, pil_kwargs={"quality": JPEG_QUALITY})
    print(f"Gespeichert: {OUT_PATH.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
