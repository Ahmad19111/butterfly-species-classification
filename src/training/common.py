"""Gemeinsame Funktionen für das Training."""
import torch
from torch.utils.data import DataLoader

from src.utils.repro import make_generator, seed_worker

MB = 1024 ** 2


def make_loader(ds, cfg, seed, shuffle):
    return DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"],
        pin_memory=True, persistent_workers=cfg["num_workers"] > 0,
        worker_init_fn=seed_worker, generator=make_generator(seed),
    )


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None,
              max_batches=None, accum_steps=1):
    """Eine Trainings- oder Evaluations-Epoche ausführen."""
    train = optimizer is not None
    n_batches = min(len(loader), max_batches) if max_batches else len(loader)
    total_loss, correct, n = 0.0, 0, 0
    if train:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                # Loss skalieren und erst nach accum_steps Batches aktualisieren.
                scaler.scale(loss / accum_steps).backward()
                if (i + 1) % accum_steps == 0 or i + 1 == n_batches:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def peak_mem_mb(device):
    if device.type != "cuda":
        return None, None
    return torch.cuda.max_memory_allocated() / MB, torch.cuda.max_memory_reserved() / MB
