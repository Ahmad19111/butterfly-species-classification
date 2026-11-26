"""Rohbilder ausrichten, nach RGB konvertieren und verkleinern."""
from pathlib import Path

import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
RAW_DIR = ROOT / CFG["raw_dir"]
OUT_DIR = ROOT / CFG["processed_dir"]
SHORT_SIDE = CFG["preprocess"]["short_side"]
QUALITY = CFG["preprocess"]["jpeg_quality"]


def process_image(src, dst):
    with Image.open(src) as im:
        # EXIF-Ausrichtung vor dem Speichern anwenden.
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if min(w, h) > SHORT_SIDE:  # verkleinern
            scale = SHORT_SIDE / min(w, h)
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        im.save(dst, "JPEG", quality=QUALITY)


def main():
    failed = []
    n_ok = 0
    for cls_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        out_cls = OUT_DIR / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        files = sorted(f for f in cls_dir.iterdir() if f.is_file())
        for f in tqdm(files, desc=cls_dir.name, leave=False):
            try:
                process_image(f, out_cls / f.name)
                n_ok += 1
            except Exception as e:
                failed.append(f"{f.relative_to(ROOT)} ({type(e).__name__}: {e})")
        print(f"{cls_dir.name}: {len(files)} Dateien")

    print(f"Verarbeitet: {n_ok}, fehlgeschlagen: {len(failed)}")
    for f in failed:
        print("  ", f)


if __name__ == "__main__":
    main()
