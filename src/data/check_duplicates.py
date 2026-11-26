"""Ähnliche Bilder in den Splits mit pHash finden."""
from itertools import combinations
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
SPLIT_DIR = ROOT / CFG["splits_dir"]
DCFG = CFG["duplicates"]
SPLITS = ("train", "val", "test")
HASH_BITS = 64  # Anzahl Bits im pHash.
OUT_CSV = ROOT / "data" / "metadata" / "duplicates_cross_split.csv"
OUT_PNG = ROOT / "data" / "metadata" / "duplicates_examples.png"


def hash_split(df):
    return phash_codes(df.path)


def phash_codes(paths):
    hashes = []
    for p in tqdm(paths, desc="pHash", leave=False):
        with Image.open(ROOT / p) as im:
            hashes.append(imagehash.phash(im, hash_size=DCFG["hash_size"]).hash.flatten())
    # 64 Bits passen in einen uint64.
    return np.packbits(np.array(hashes), axis=1).view(np.uint64).ravel()


def find_pairs(df_a, h_a, df_b, h_b):
    dist = np.bitwise_count(h_a[:, None] ^ h_b[None, :])
    ia, ib = np.nonzero(dist <= DCFG["max_distance"])
    return pd.DataFrame({
        "split_a": df_a.split.values[ia], "path_a": df_a.path.values[ia],
        "class_a": df_a.class_name.values[ia],
        "split_b": df_b.split.values[ib], "path_b": df_b.path.values[ib],
        "class_b": df_b.class_name.values[ib],
        "distance": dist[ia, ib],
    })


def plot_examples(pairs):
    n = min(DCFG["n_examples"], len(pairs))
    if n == 0:
        return
    ex = pairs.sort_values("distance").head(n)
    fig, axes = plt.subplots(n, 2, figsize=(4.4, 2.2 * n))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(ex.itertuples()):
        for c, (path, split, cls) in enumerate([(row.path_a, row.split_a, row.class_a),
                                                 (row.path_b, row.split_b, row.class_b)]):
            ax = axes[r, c]
            with Image.open(ROOT / path) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{split} | {cls}\n{Path(path).name}", fontsize=7)
            ax.axis("off")
        axes[r, 0].text(1.05, -0.08, f"Distanz {row.distance}", transform=axes[r, 0].transAxes,
                        ha="center", fontsize=7)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)


def main():
    assert DCFG["hash_size"] ** 2 == HASH_BITS, "hash_size muss 8 sein (64-Bit-Vergleich)"
    dfs, hashes = {}, {}
    for s in SPLITS:
        dfs[s] = pd.read_csv(SPLIT_DIR / f"{s}.csv").assign(split=s)
        hashes[s] = hash_split(dfs[s])

    pairs = pd.concat([find_pairs(dfs[a], hashes[a], dfs[b], hashes[b])
                       for a, b in combinations(SPLITS, 2)], ignore_index=True)
    pairs["same_class"] = pairs.class_a == pairs.class_b
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pairs.sort_values(["distance", "split_a", "split_b"]).to_csv(OUT_CSV, index=False)

    print(f"Schwellwert: Hamming-Distanz <= {DCFG['max_distance']} (von {HASH_BITS} Bit)")
    for a, b in combinations(SPLITS, 2):
        sub = pairs[(pairs.split_a == a) & (pairs.split_b == b)]
        print(f"{a}-{b}: {len(sub)} Paare, {sub.path_b.nunique()} betroffene {b}-Bilder, "
              f"davon andere Klasse: {(~sub.same_class).sum()}")
    if len(pairs):
        print("Paare pro Distanz:", pairs.distance.value_counts().sort_index().to_dict())
    plot_examples(pairs)
    print(f"Gespeichert: {OUT_CSV.relative_to(ROOT).as_posix()}"
          + (f", {OUT_PNG.relative_to(ROOT).as_posix()}" if len(pairs) else ""))


if __name__ == "__main__":
    main()
