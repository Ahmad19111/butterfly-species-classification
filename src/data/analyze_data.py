"""Bildstatistik pro Klasse erstellen."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
RAW_DIR = ROOT / CFG["raw_dir"]
SHORT_SIDE = CFG["preprocess"]["short_side"]
OUT_CSV = ROOT / "data" / "metadata" / "data_stats.csv"

EXIF_ORIENTATION = 274
ORIENTATION_NORMAL = 1
# EXIF-Werte mit gedrehter Bildausrichtung.
ORIENTATIONS_ROTATED = (3, 5, 6, 7, 8)


def analyze_class(cls_dir):
    ws, hs, bad = [], [], []
    fmts, modes, exts, orient = Counter(), Counter(), Counter(), Counter()
    files = sorted(f for f in cls_dir.iterdir() if f.is_file())
    for f in files:
        exts[f.suffix.lower()] += 1
        try:
            with Image.open(f) as im:
                im.verify()
            # Nach verify() das Bild erneut öffnen.
            with Image.open(f) as im:
                im.load()
                ws.append(im.width)
                hs.append(im.height)
                fmts[im.format] += 1
                modes[im.mode] += 1
                orient[im.getexif().get(EXIF_ORIENTATION, ORIENTATION_NORMAL)] += 1
        except Exception as e:
            bad.append(f"{f.name} ({type(e).__name__})")

    ws, hs = np.array(ws), np.array(hs)
    short = np.minimum(ws, hs)
    return {
        "class_name": cls_dir.name,
        "n_files": len(files),
        "n_readable": len(ws),
        "width_min": ws.min(), "width_median": int(np.median(ws)), "width_max": ws.max(),
        "height_min": hs.min(), "height_median": int(np.median(hs)), "height_max": hs.max(),
        "short_side_min": short.min(), "short_side_median": int(np.median(short)),
        f"n_short_lt_{SHORT_SIDE}": int((short < SHORT_SIDE).sum()),
        "n_exif_rotated": sum(orient[k] for k in ORIENTATIONS_ROTATED),
        "extensions": ";".join(f"{k}:{v}" for k, v in exts.items()),
        "formats": ";".join(f"{k}:{v}" for k, v in fmts.items()),
        "modes": ";".join(f"{k}:{v}" for k, v in modes.items()),
        "n_corrupt": len(bad),
        "corrupt_files": ";".join(bad),
    }


def main():
    rows = [analyze_class(d) for d in sorted(RAW_DIR.iterdir()) if d.is_dir()]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    cols = ["class_name", "n_files", "n_readable", "width_median", "height_median",
            "short_side_min", f"n_short_lt_{SHORT_SIDE}", "n_exif_rotated", "modes", "n_corrupt"]
    print(df[cols].to_string(index=False))
    print(f"Gesamt: {df.n_files.sum()} Dateien, {df.n_readable.sum()} lesbar")
    print(f"Gespeichert: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
