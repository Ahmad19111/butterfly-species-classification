"""ResNet50 in zwei Phasen feinabstimmen."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from src.data.dataset import ButterflyDataset
from src.data.transforms import build_transforms, norm_from_torchvision
from src.models.cnn import build_resnet50, set_backbone_trainable
from src.training.common import make_loader, peak_mem_mb, run_epoch
from src.utils.repro import set_seed

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn_resnet50.yaml")
    parser.add_argument("--debug", action="store_true", help="kurzer Testlauf")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tcfg["amp"] and device.type == "cuda"

    train_ds = ButterflyDataset("train")
    n_classes = len(train_ds.class_names)
    model, weights = build_resnet50(n_classes, cfg["model"]["weights"])
    model.to(device)

    mean, std = norm_from_torchvision(weights)
    train_ds.transform = build_transforms(True, mean, std, cfg["transforms"])
    val_ds = ButterflyDataset("val", build_transforms(False, mean, std, cfg["transforms"]))
    train_loader = make_loader(train_ds, tcfg, cfg["seed"], shuffle=True)
    val_loader = make_loader(val_ds, tcfg, cfg["seed"], shuffle=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Klassen: {n_classes}, Gerät: {device}")

    suffix = "_debug" if args.debug else ""
    ckpt_path = ROOT / cfg["output"]["checkpoint_dir"] / f"{cfg['name']}{suffix}_best.pt"
    res_path = ROOT / cfg["output"]["results_dir"] / f"{cfg['name']}{suffix}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    max_batches = cfg["debug"]["max_batches"] if args.debug else None

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history, phases_info = [], {}
    best_acc, best_epoch, epoch_global = -1.0, None, 0
    t_start = time.perf_counter()

    for phase, frozen in (("phase_a", True), ("phase_b", False)):
        pcfg = tcfg[phase]
        epochs = cfg["debug"]["epochs"] if args.debug else pcfg["epochs"]
        set_backbone_trainable(model, not frozen)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimizer = torch.optim.AdamW(params, lr=pcfg["lr"], weight_decay=tcfg["weight_decay"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"{phase}: trainierbare Parameter {n_trainable:,}, lr {pcfg['lr']}, Epochen {epochs}")

        train_time = 0.0
        for _ in range(epochs):
            epoch_global += 1
            # BatchNorm-Statistiken des eingefrorenen Backbones nicht aktualisieren.
            model.train(not frozen)
            t0 = time.perf_counter()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, use_amp,
                                        optimizer, scaler, max_batches)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ep_time = time.perf_counter() - t0
            train_time += ep_time

            model.eval()
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp,
                                        max_batches=max_batches)
            improved = va_acc > best_acc
            if improved:
                best_acc, best_epoch = va_acc, epoch_global
                torch.save(model.state_dict(), ckpt_path)
            print(f"[{phase}] Epoche {epoch_global}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {ep_time:.0f}s{' *' if improved else ''}")
            history.append({"epoch": epoch_global, "phase": phase, "train_loss": tr_loss,
                            "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc,
                            "train_time_s": ep_time})

        peak_alloc, peak_reserved = peak_mem_mb(device)
        phases_info[phase] = {"trainable_params": n_trainable, "epochs": epochs, "lr": pcfg["lr"],
                              "train_time_s": train_time, "peak_gpu_mem_allocated_mb": peak_alloc,
                              "peak_gpu_mem_reserved_mb": peak_reserved}
        print(f"{phase}: Trainingszeit {train_time:.0f}s, max. GPU-Speicher {peak_alloc:.0f} MB "
              f"(reserviert {peak_reserved:.0f} MB)")

    total_time = time.perf_counter() - t_start
    results = {
        "name": cfg["name"], "debug": args.debug, "config": cfg,
        "total_params": sum(p.numel() for p in model.parameters()),
        "phases": phases_info, "total_wall_time_s": total_time,
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
