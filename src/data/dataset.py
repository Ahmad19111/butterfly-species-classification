"""Dataset für die Split-CSV-Dateien."""
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
SPLIT_DIR = ROOT / DATA_CFG["splits_dir"]


class ButterflyDataset(Dataset):
    def __init__(self, split, transform=None, split_dir=None):
        # Optionalen Split-Ordner verwenden, sonst den Pfad aus data.yaml.
        df = pd.read_csv(Path(split_dir or SPLIT_DIR) / f"{split}.csv")
        self.paths = [ROOT / p for p in df.path]
        self.labels = df.label.tolist()
        self.class_names = df.drop_duplicates("label").sort_values("label").class_name.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]
