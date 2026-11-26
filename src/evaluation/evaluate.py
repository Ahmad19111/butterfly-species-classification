"""Modelle auswerten und Metriken als JSON und CSV speichern."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.data.dataset import ButterflyDataset
from src.data.transforms import build_transforms, norm_from_timm, norm_from_torchvision
from src.models.cnn import build_resnet50
from src.models.vit import build_vit, load_lora
from src.training.common import make_loader
from src.utils.repro import set_seed

ROOT = Path(__file__).resolve().parents[2]
MIN_PER_S = 1 / 60
METRICS = ["accuracy", "precision", "recall", "f1"]
CI_METRICS = ["accuracy", "f1"]


def load_yaml(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def load_model(run, mcfg, n_classes, ckpt, device):
    if run["kind"] == "cnn":
        model, weights = build_resnet50(n_classes, mcfg["model"]["weights"])
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        mean, std = norm_from_torchvision(weights)
    elif run["kind"] == "vit_lp":
        model = build_vit(mcfg["model"]["timm_name"], n_classes)
        model.get_classifier().load_state_dict(torch.load(ckpt, map_location="cpu"))
        mean, std = norm_from_timm(model)
    else:
        base = build_vit(mcfg["model"]["timm_name"], n_classes)
        mean, std = norm_from_timm(base)
        model = load_lora(base, ckpt)
    return model.to(device).eval(), mean, std


def predict(model, loader, device, use_amp):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x.to(device, non_blocking=True))
            y_pred.append(out.argmax(1).cpu())
            y_true.append(y)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def bootstrap_ci(y_true, y_pred, n_resamples, level, seed):
    """Bootstrap-Konfidenzintervalle für Accuracy und Macro-F1 berechnen."""
    # Gleicher Seed und gleiche Testgröße liefern gepaarte Resamples.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n_resamples, len(y_true)))
    labels = np.unique(y_true)
    acc = (y_true[idx] == y_pred[idx]).mean(axis=1)
    f1 = np.array([f1_score(y_true[i], y_pred[i], labels=labels, average="macro", zero_division=0)
                   for i in idx])
    q = [(1 - level) / 2 * 100, (1 + level) / 2 * 100]
    out = {}
    for m, vals in (("accuracy", acc), ("f1", f1)):
        out[f"{m}_lo"], out[f"{m}_hi"] = np.percentile(vals, q)
    return out


def efficiency(res):
    # Beim CNN zählt die Phase mit den meisten trainierbaren Parametern.
    ph = res["phases"].values()
    mems = [p["peak_gpu_mem_allocated_mb"] for p in ph if p["peak_gpu_mem_allocated_mb"] is not None]
    return {
        "trainable_params": max(p["trainable_params"] for p in ph),
        "train_time_min": sum(p["train_time_s"] for p in ph) * MIN_PER_S,
        "peak_gpu_mem_mb": max(mems) if mems else None,
        "epochs": sum(p["epochs"] for p in ph),
        "best_val_acc": res["best_val_acc"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--device", default=None, help="z. B. cpu; Standard: cuda falls verfügbar")
    args = parser.parse_args()

    ecfg = load_yaml(args.config)
    ocfg, bcfg = ecfg["output"], ecfg["bootstrap"]
    set_seed(ecfg["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = ecfg["amp"] and device.type == "cuda"
    res_dir = ROOT / ocfg["results_dir"]

    class_names = ButterflyDataset("test").class_names
    rows = []
    for run in ecfg["runs"]:
        res_path = res_dir / f"{run['name']}.json"
        if not res_path.exists():
            print(f"Übersprungen: {run['name']} (kein Ergebnis-JSON)")
            continue
        res = json.loads(res_path.read_text(encoding="utf-8"))
        ckpt = ROOT / res["checkpoint"]
        if not ckpt.exists():
            print(f"Übersprungen: {run['name']} (Checkpoint fehlt: {res['checkpoint']})")
            continue

        mcfg = load_yaml(run["config"])
        model, mean, std = load_model(run, mcfg, len(class_names), ckpt, device)
        split_dir = ROOT / run["splits_dir"] if "splits_dir" in run else None
        test_ds = ButterflyDataset("test", build_transforms(False, mean, std, mcfg["transforms"]), split_dir)
        loader = make_loader(test_ds, ecfg, ecfg["seed"], shuffle=False)
        y_true, y_pred = predict(model, loader, device, use_amp)
        del model, loader  # DataLoader-Worker freigeben.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        rows.append({
            "name": run["name"], "label": run["label"], "kind": run["kind"], "rank": run.get("rank"),
            "accuracy": accuracy_score(y_true, y_pred), "precision": prec, "recall": rec, "f1": f1,
            **bootstrap_ci(y_true, y_pred, bcfg["n_resamples"], bcfg["level"], ecfg["seed"]),
            **efficiency(res),
        })

        pd.DataFrame({"path": [p.relative_to(ROOT).as_posix() for p in test_ds.paths],
                      "label": y_true, "pred": y_pred}).to_csv(
            res_dir / f"test_predictions_{run['name']}.csv", index=False)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True,
                                       zero_division=0)
        (res_dir / f"test_metrics_{run['name']}.json").write_text(
            json.dumps({"report": report, "confusion_matrix": cm.tolist(), "class_names": class_names},
                       indent=2), encoding="utf-8")
        r = rows[-1]
        print(f"{run['label']}: Acc {r['accuracy']:.4f} [{r['accuracy_lo']:.4f}, {r['accuracy_hi']:.4f}], "
              f"F1 {f1:.4f} [{r['f1_lo']:.4f}, {r['f1_hi']:.4f}]")

    if not rows:
        print("Keine Modelle gefunden.")
        return
    table = pd.DataFrame(rows)
    ci_cols = [f"{m}_{b}" for m in CI_METRICS for b in ("lo", "hi")]
    cols = ["name", "label", "rank", *METRICS, *ci_cols, "trainable_params", "train_time_min",
            "peak_gpu_mem_mb", "epochs", "best_val_acc"]
    table[cols].to_csv(ROOT / ocfg["table_csv"], index=False)

    print(table[cols].to_string(index=False))
    print(f"Gespeichert: {ocfg['table_csv']} und Testmetriken/Vorhersagen in {res_dir}/")


if __name__ == "__main__":
    main()
