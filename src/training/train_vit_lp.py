"""ViT Linear Probing mit zwischengespeicherten Features."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import SPLIT_DIR, ButterflyDataset
from src.data.transforms import build_transforms, norm_from_timm
from src.models.vit import build_vit, extract_features, freeze_backbone
from src.training.common import make_loader, peak_mem_mb, run_epoch
from src.utils.repro import make_generator, set_seed

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def compute_features(model, loader, device, use_amp, max_batches):
    feats, labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                f = extract_features(model, x.to(device, non_blocking=True))
            feats.append(f.float().cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def load_or_extract(model, split, cfg, mean, std, device, cache_dir, max_batches):
    """Features aus dem Cache laden oder neu berechnen."""
    fcfg = cfg["features"]
    augment = split == "train" and fcfg["train_augment"]
    # Cache nur bei passendem Modell, Transform und Split verwenden.
    split_md5 = hashlib.md5((SPLIT_DIR / f"{split}.csv").read_bytes()).hexdigest()
    meta = {"model": cfg["model"]["timm_name"], "transforms": cfg["transforms"],
            "augment": augment, "max_batches": max_batches, "split_md5": split_md5}
    path = cache_dir / f"{split}.pt"
    if path.exists():
        cached = torch.load(path)
        if cached["meta"] == meta:
            return cached, True

    ds = ButterflyDataset(split, build_transforms(augment, mean, std, cfg["transforms"]))
    loader = make_loader(ds, fcfg, cfg["seed"], shuffle=False)
    use_amp = fcfg["amp"] and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    feats, labels = compute_features(model, loader, device, use_amp, max_batches)
    if device.type == "cuda":
        torch.cuda.synchronize()
    alloc, reserved = peak_mem_mb(device)
    cached = {"meta": meta, "feats": feats, "labels": labels, "time_s": time.perf_counter() - t0,
              "peak_gpu_mem_allocated_mb": alloc, "peak_gpu_mem_reserved_mb": reserved}
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cached, path)
    return cached, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_linear_probe.yaml")
    parser.add_argument("--debug", action="store_true", help="kurzer Testlauf")
    parser.add_argument("--device", default=None, help="z. B. cpu; Standard: cuda falls verfügbar")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = tcfg["amp"] and device.type == "cuda"
    suffix = "_debug" if args.debug else ""
    max_batches = cfg["debug"]["max_batches"] if args.debug else None

    class_names = ButterflyDataset("train").class_names
    model = build_vit(cfg["model"]["timm_name"], len(class_names))
    freeze_backbone(model)
    model.to(device).eval()
    mean, std = norm_from_timm(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modell: {cfg['model']['timm_name']}, mean {mean}, std {std}, Gerät: {device}")

    t_start = time.perf_counter()
    cache_dir = ROOT / cfg["features"]["cache_dir"] / f"{cfg['model']['timm_name']}{suffix}"
    data = {}
    for split in SPLITS:
        data[split], from_cache = load_or_extract(model, split, cfg, mean, std, device,
                                                  cache_dir, max_batches)
        print(f"Features {split}: {tuple(data[split]['feats'].shape)}, "
              f"{data[split]['time_s']:.0f}s{' (Cache)' if from_cache else ''}")
    tr = data["train"]

    head = model.get_classifier()
    gen = make_generator(cfg["seed"])
    train_loader = DataLoader(TensorDataset(tr["feats"], tr["labels"]),
                              batch_size=tcfg["batch_size"], shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(data["val"]["feats"], data["val"]["labels"]),
                            batch_size=tcfg["batch_size"])

    ckpt_path = ROOT / cfg["output"]["checkpoint_dir"] / f"{cfg['name']}{suffix}_best_head.pt"
    res_path = ROOT / cfg["output"]["results_dir"] / f"{cfg['name']}{suffix}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer = torch.optim.AdamW(head.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    epochs = cfg["debug"]["epochs"] if args.debug else tcfg["epochs"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"head_training: trainierbare Parameter {n_trainable:,}, lr {tcfg['lr']}, Epochen {epochs}")

    history, best_acc, best_epoch, train_time = [], -1.0, None, 0.0
    for epoch in range(1, epochs + 1):
        head.train()
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(head, train_loader, criterion, device, use_amp, optimizer, scaler)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        train_time += ep_time
        head.eval()
        va_loss, va_acc = run_epoch(head, val_loader, criterion, device, use_amp)
        improved = va_acc > best_acc
        if improved:
            best_acc, best_epoch = va_acc, epoch
            torch.save(head.state_dict(), ckpt_path)
        print(f"[head_training] Epoche {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {ep_time:.1f}s{' *' if improved else ''}")
        history.append({"epoch": epoch, "phase": "head_training", "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                        "train_time_s": ep_time})
    alloc, reserved = peak_mem_mb(device)

    # Feature-Extraktion des Trainingssplits zur Trainingszeit zählen.
    phases = {
        "feature_extraction": {
            "trainable_params": 0, "epochs": 1, "lr": None, "train_time_s": tr["time_s"],
            "peak_gpu_mem_allocated_mb": tr["peak_gpu_mem_allocated_mb"],
            "peak_gpu_mem_reserved_mb": tr["peak_gpu_mem_reserved_mb"],
            "time_per_split_s": {s: data[s]["time_s"] for s in SPLITS},
        },
        "head_training": {
            "trainable_params": n_trainable, "epochs": epochs, "lr": tcfg["lr"],
            "train_time_s": train_time, "peak_gpu_mem_allocated_mb": alloc,
            "peak_gpu_mem_reserved_mb": reserved,
        },
    }
    total_time = time.perf_counter() - t_start
    results = {
        "name": cfg["name"], "debug": args.debug, "config": cfg,
        "total_params": sum(p.numel() for p in model.parameters()),
        "phases": phases, "total_wall_time_s": total_time,
        "best_val_acc": best_acc, "best_epoch": best_epoch,
        "checkpoint": ckpt_path.relative_to(ROOT).as_posix(), "history": history,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Beste Val-Acc {best_acc:.4f} (Epoche {best_epoch}), Gesamtzeit {total_time:.0f}s")
    print(f"Gespeichert: {res_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
