"""McNemar-Tests für Modelle auf demselben Test-Set."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]


def holm(pvals):
    # Holm-Bonferroni-Korrektur für alle Paarvergleiche.
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate.yaml")
    args = parser.parse_args()

    ecfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    mcfg, res_dir = ecfg["mcnemar"], ROOT / ecfg["output"]["results_dir"]
    # Läufe mit anderem Test-Set sind nicht gepaart vergleichbar.
    runs = [r for r in ecfg["runs"] if "splits_dir" not in r]

    preds, ref = {}, None
    for run in runs:
        path = res_dir / f"test_predictions_{run['name']}.csv"
        if not path.exists():
            print(f"Übersprungen: {run['name']} (keine Test-Predictions)")
            continue
        df = pd.read_csv(path)
        if ref is None:
            ref = df
        # Die Vorhersagen müssen dieselben Bilder in derselben Reihenfolge enthalten.
        assert df.path.equals(ref.path) and df.label.equals(ref.label), run["name"]
        preds[run["name"]] = (run["label"], (df.pred == df.label).to_numpy())

    rows = []
    for a, b in combinations(preds, 2):
        (la, ca), (lb, cb) = preds[a], preds[b]
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())  # Nur A / nur B richtig.
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        rows.append({"model_a": a, "model_b": b, "label_a": la, "label_b": lb,
                     "acc_a": ca.mean(), "acc_b": cb.mean(), "only_a_correct": n01,
                     "only_b_correct": n10, "p_value": p})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p_value)
    table["significant"] = table.p_holm < mcfg["alpha"]

    out_csv = ROOT / mcfg["table_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)

    print(table[["label_a", "label_b", "only_a_correct", "only_b_correct", "p_value", "p_holm",
                 "significant"]].to_string(index=False))
    print(f"Gespeichert: {mcfg['table_csv']}")


if __name__ == "__main__":
    main()
