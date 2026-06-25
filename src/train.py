"""Training script for 1D U-Net spectrum transformer."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import load_spectrum_file
from src.dataset import FixedMixtureSet, SyntheticMixtureDataset
from src.losses import SpectrumLoss
from src.model import UNet1D
from src.preprocessing import Preprocessor, build_preprocessors
from src.synth import SynthGenerator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mono_spectra(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Load all training files and return monoenergetic spectra arrays.

    Returns
    -------
    energy_grid : (N,)
    mono_A      : (K, N)  SiC_SBD
    mono_B      : (K, N)  TEPC
    energies_MeV: list of K floats
    """
    import glob

    # Check if processed data exists
    processed_dir = Path("data/processed")
    npz_path = processed_dir / "mono_spectra.npz"
    grid_path = processed_dir / "energy_grid.npy"

    if npz_path.exists() and grid_path.exists():
        energy_grid = np.load(str(grid_path))
        data = np.load(str(npz_path))
        return energy_grid, data["mono_A"], data["mono_B"], list(data["energies_MeV"])

    # Load from raw files
    files = sorted(glob.glob(cfg.data.raw_glob))
    if not files:
        raise FileNotFoundError(f"No files found matching {cfg.data.raw_glob}")

    all_A = []
    all_B = []
    all_E = []
    energy_grid = None

    for f in files:
        spec = load_spectrum_file(f, skiprows=cfg.data.sheet_skiprows)
        if energy_grid is None:
            energy_grid = spec["energy"]
        if spec["energy_MeV"] is not None:
            all_A.append(spec["SiC_SBD"])
            all_B.append(spec["TEPC"])
            all_E.append(spec["energy_MeV"])

    mono_A = np.stack(all_A)  # (K, N)
    mono_B = np.stack(all_B)
    return energy_grid, mono_A, mono_B, all_E


def build_generators(
    energy_grid: np.ndarray,
    mono_A: np.ndarray,
    mono_B: np.ndarray,
    energies_MeV: list[float],
    heldout: list[float],
    cfg,
) -> tuple[SynthGenerator, SynthGenerator]:
    """Build train and val/test generators.

    Training uses non-heldout energies; val/test uses heldout energies.
    """
    heldout_set = set(heldout)
    train_mask = [e not in heldout_set for e in energies_MeV]
    heldout_mask = [e in heldout_set for e in energies_MeV]

    train_idx = [i for i, m in enumerate(train_mask) if m]
    held_idx = [i for i, m in enumerate(heldout_mask) if m]

    families = list(cfg.synth.mixture_families)

    train_gen = SynthGenerator(
        mono_A[train_idx],
        mono_B[train_idx],
        np.array([energies_MeV[i] for i in train_idx]),
        families=families,
        poisson_noise=cfg.synth.poisson_noise,
        gcr_powerlaw_index=cfg.synth.gcr_powerlaw_index,
    )

    if held_idx:
        val_gen = SynthGenerator(
            mono_A[held_idx],
            mono_B[held_idx],
            np.array([energies_MeV[i] for i in held_idx]),
            families=families,
            poisson_noise=cfg.synth.poisson_noise,
            gcr_powerlaw_index=cfg.synth.gcr_powerlaw_index,
        )
    else:
        # Fall back to training energies if no heldout available
        val_gen = train_gen

    return train_gen, val_gen


def train(cfg, fast_dev_run: bool = False) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    energy_grid, mono_A, mono_B, energies_MeV = load_mono_spectra(cfg)
    print(f"Loaded {len(energies_MeV)} energies: {sorted(energies_MeV)}")

    # Normalize monoenergetic spectra
    for i in range(len(mono_A)):
        s = mono_A[i].sum()
        if s > 0:
            mono_A[i] = mono_A[i] / s
        s = mono_B[i].sum()
        if s > 0:
            mono_B[i] = mono_B[i] / s

    heldout = list(cfg.data.heldout_energies_MeV)
    train_gen, val_gen = build_generators(
        energy_grid, mono_A, mono_B, energies_MeV, heldout, cfg
    )

    input_pre, target_pre = build_preprocessors(cfg)

    samples_per_epoch = 2 if fast_dev_run else cfg.train.samples_per_epoch
    val_n = 4 if fast_dev_run else cfg.train.val_mixtures
    test_n = 4 if fast_dev_run else cfg.train.test_mixtures

    train_ds = SyntheticMixtureDataset(
        train_gen, input_pre, target_pre,
        samples_per_epoch=samples_per_epoch, base_seed=cfg.seed,
    )
    val_ds = FixedMixtureSet(
        val_gen, input_pre, target_pre, n_samples=val_n, seed=cfg.seed + 1
    )
    test_ds = FixedMixtureSet(
        val_gen, input_pre, target_pre, n_samples=test_n, seed=cfg.seed + 2
    )

    batch_size = min(cfg.train.batch_size, samples_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = UNet1D(
        in_ch=cfg.model.in_ch,
        out_ch=1,
        base=cfg.model.base_channels,
        depth=cfg.model.depth,
        head=cfg.model.head,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    criterion = SpectrumLoss(w_mse=cfg.loss.w_mse, w_emd=cfg.loss.w_emd)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    epochs = 1 if fast_dev_run else cfg.train.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # AMP
    use_amp = cfg.train.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Logging
    run_dir = Path("runs") / f"baseline_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(str(run_dir))
    OmegaConf.save(cfg, str(run_dir / "config.yaml"))

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        batches_done = 0

        for batch in train_loader:
            if fast_dev_run and batches_done >= 2:
                break

            x = batch["input"].to(device)
            y = batch["target"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss, loss_dict = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            batches_done += 1

        scheduler.step()
        avg_train = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    loss, _ = criterion(pred, y)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Loss/val", avg_val, epoch)
        writer.add_scalar("LR", lr, epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"train={avg_train:.4f} val={avg_val:.4f} lr={lr:.2e}"
            )

        # Checkpoint
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": avg_val,
            "cfg": OmegaConf.to_container(cfg),
        }
        torch.save(state, str(ckpt_dir / "last.pt"))

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(state, str(ckpt_dir / "best.pt"))

    writer.close()
    print(f"Training done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}/best.pt, {ckpt_dir}/last.pt")
    print(f"TensorBoard logs: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 1D U-Net spectrum transformer")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run 2 batches x 1 epoch")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
