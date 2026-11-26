"""Train-, Val- und Test-Splits nach Bildgruppen erstellen."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.check_duplicates import phash_codes

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
PROC_DIR = ROOT / CFG["processed_dir"]
SPLIT_DIR = ROOT / CFG["splits_dir"]
RATIOS = CFG["splits"]
SEED = CFG["seed"]
DUP_MAX_DIST = CFG["duplicates"]["max_distance"]
ID_PATTERN = r"sighting_(\d+)"
# Kleine Splits zuerst füllen; übrige Gruppen kommen in train.
FILL_ORDER = ("test", "val")


def assign_groups(df, max_gap):
    """Nahe Sighting-IDs und pHash-Duplikate zu Gruppen verbinden."""
    df = df.sort_values(["label", "sighting_id"]).reset_index(drop=True)
    # Nahe Sighting-IDs derselben Klasse verbinden.
    nb = (df.sighting_id.diff() <= max_gap) & (df.label == df.label.shift())
    rows, cols = list(np.nonzero(nb.values)[0] - 1), list(np.nonzero(nb.values)[0])
    # Ähnliche Fotos unabhängig von ihrer Sighting-ID verbinden.
    codes = phash_codes(df.path)
    for _, idx in df.groupby("label").groups.items():
        idx = np.asarray(idx)
        dist = np.bitwise_count(codes[idx][:, None] ^ codes[idx][None, :])
        a, b = np.nonzero(np.triu(dist <= DUP_MAX_DIST, k=1))
        rows += list(idx[a])
        cols += list(idx[b])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(df)))
    _, df["group"] = connected_components(graph, directed=False)
    return df


def group_split(df):
    """Gruppen pro Klasse mischen und den Splits zuweisen."""
    rng = np.random.default_rng(SEED)
    split_of = {}
    for _, cls_df in df.groupby("label"):
        sizes = cls_df.groupby("group").size()
        filled = dict.fromkeys(FILL_ORDER, 0)
        for g in rng.permutation(sizes.index):
            s = next((s for s in FILL_ORDER if filled[s] < RATIOS[s] * len(cls_df)), "train")
            if s != "train":
                filled[s] += sizes[g]
            split_of[g] = s
    return df.group.map(split_of)


def main():
    class_names = sorted(d.name for d in PROC_DIR.iterdir() if d.is_dir())
    rows = [
        {"path": f.relative_to(ROOT).as_posix(), "label": label, "class_name": name}
        for label, name in enumerate(class_names)
        for f in sorted((PROC_DIR / name).iterdir()) if f.is_file()
    ]
    df = pd.DataFrame(rows)
    df["sighting_id"] = df.path.str.extract(ID_PATTERN, expand=False).astype(int)
    df = assign_groups(df, RATIOS["group_max_id_gap"])
    df["split"] = group_split(df)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "class_name", "group"]
    for name in ("train", "val", "test"):
        df[df.split == name][cols].sort_values("path").to_csv(SPLIT_DIR / f"{name}.csv", index=False)

    counts = pd.crosstab(df.class_name, df.split)[["train", "val", "test"]]
    counts.loc["Summe"] = counts.sum()
    counts["gesamt"] = counts.sum(axis=1)
    print(f"Gruppen: {df.group.nunique()} (max. ID-Abstand {RATIOS['group_max_id_gap']}), "
          f"größte Gruppe: {df.group.value_counts().max()} Bilder")
    print(counts.to_string())


if __name__ == "__main__":
    main()
