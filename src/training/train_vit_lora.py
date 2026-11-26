"""ViT mit LoRA auf qkv feinabstimmen."""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from src.data.dataset import ButterflyDataset
from src.data.transforms import build_transforms, norm_from_timm
from src.models.vit import add_lora, build_vit
from src.training.common import make_loader, peak_mem_mb, run_epoch
from src.utils.repro import set_seed

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vit_lora.yaml")
    parser.add_argument("--rank", type=int, default=None, help="überschreibt lora.rank")
    parser.add_argument("--debug", action="store_true", help="kurzer Testlauf")
    parser.add_argument("--device", default=None, help="z. B. cpu; Standard: cuda falls verfügbar")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    lcfg, tcfg = cfg["lora"], cfg["train"]
    if args.rank is not None:
        lcfg["rank"] = args.rank
    rank = lcfg["rank"]
    alpha = lcfg["alpha_factor"] * rank
    run_name = f"{cfg['name']}_r{rank}"

    set_seed(cfg["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = tcfg["amp"] and device.type == "cuda"

    train_ds = ButterflyDataset("train")
    base = build_vit(cfg["model"]["timm_name"], len(train_ds.class_names))
    mean, std = norm_from_timm(base)
    model = add_lora(base, rank, alpha, lcfg["dropout"], lcfg["target_modules"], lcfg["modules_to_save"])
    model.to(device)

    train_ds.transform = build_transforms(True, mean, std, cfg["transforms"])
    val_ds = ButterflyDataset("val", build_transforms(False, mean, std, cfg["transforms"]))
    train_loader = make_loader(train_ds, tcfg, cfg["seed"], shuffle=True)
    val_loader = make_loader(val_ds, tcfg, cfg["seed"], shuffle=False)

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{run_name}: alpha {alpha}, trainierbar {n_trainable:,} (LoRA {n_lora:,}, Kopf "
          f"{n_trainable - n_lora:,}) = {100 * n_trainable / n_total:.2f} % von {n_total:,}, Gerät: {device}")

    suffix = "_debug" if args.debug else ""
    ckpt_dir = ROOT / cfg["output"]["checkpoint_dir"] / f"{run_name}{suffix}_best"
    res_path = ROOT / cfg["output"]["results_dir"] / f"{run_name}{suffix}.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    max_batches = cfg["debug"]["max_batches"] if args.debug else None
    epochs = cfg["debug"]["epochs"] if args.debug else tcfg["epochs"]

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best_acc, best_epoch, train_time = [], -1.0, None, 0.0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, use_amp, optimizer,
                                    scaler, max_batches, tcfg["accum_steps"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.perf_counter() - t0
        train_time += ep_time

        model.eval()
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp,
                                    max_batches=max_batches)
        improved = va_acc > best_acc
        if improved:
            best_acc, best_epoch = va_acc, epoch
            # Nur LoRA-Gewichte und Kopf speichern.
            model.save_pretrained(ckpt_dir)
        print(f"[lora] Epoche {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {ep_time:.0f}s{' *' if improved else ''}")
        history.append({"epoch": epoch, "phase": "lora", "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "train_time_s": ep_time})

    alloc, reserved = peak_mem_mb(device)
    phases = {"lora": {"trainable_params": n_trainable, "lora_params": n_lora, "epochs": epochs,
                       "lr": tcfg["lr"], "train_time_s": train_time,
                       "peak_gpu_mem_allocated_mb": alloc, "peak_gpu_mem_reserved_mb": reserved}}
    print(f"lora: Trainingszeit {train_time:.0f}s, max. GPU-Speicher {alloc} MB (reserviert {reserved} MB)")

    total_time = time.perf_counter() - t_start
    results = {
        "name": run_name, "debug": args.debug, "config": cfg,
        "rank": rank, "alpha": alpha, "total_params": n_total,
        "phases": phases, "total_wall_time_s": total_time,
        "best_val_acc": best_acc, "best_epoch": best_epoch,
        "checkpoint": ckpt_dir.relative_to(ROOT).as_posix(), "history": history,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
    }
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Beste Val-Acc {best_acc:.4f} (Epoche {best_epoch}), Gesamtzeit {total_time:.0f}s")
    print(f"Gespeichert: {res_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
