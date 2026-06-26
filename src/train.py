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
from src.losses import (
    SpectrumLoss,
    build_region_mask,
    build_region_weight,
    emd_1d,
    make_bin_dist,
)
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

    dirichlet_alpha_choices = list(cfg.synth.get("dirichlet_alpha_choices", [0.3, 1.0, 3.0]))
    sparse_k_range_cfg = cfg.synth.get("sparse_k_range", [2, 4])
    sparse_k_range = tuple(sparse_k_range_cfg)
    poisson_counts_range_cfg = cfg.synth.get("poisson_counts_range", [1000, 100000])
    poisson_counts_range = tuple(poisson_counts_range_cfg)
    powerlaw_alpha_range_cfg = cfg.synth.get("powerlaw_alpha_range", [-3.0, 0.0])
    powerlaw_alpha_range = tuple(powerlaw_alpha_range_cfg)

    train_gen = SynthGenerator(
        mono_A[train_idx],
        mono_B[train_idx],
        np.array([energies_MeV[i] for i in train_idx]),
        families=families,
        poisson_noise=cfg.synth.poisson_noise,
        dirichlet_alpha_choices=dirichlet_alpha_choices,
        sparse_k_range=sparse_k_range,
        poisson_counts_range=poisson_counts_range,
        powerlaw_alpha_range=powerlaw_alpha_range,
    )

    if held_idx:
        val_gen = SynthGenerator(
            mono_A[held_idx],
            mono_B[held_idx],
            np.array([energies_MeV[i] for i in held_idx]),
            families=families,
            poisson_noise=cfg.synth.poisson_noise,
            dirichlet_alpha_choices=dirichlet_alpha_choices,
            sparse_k_range=sparse_k_range,
            poisson_counts_range=poisson_counts_range,
            powerlaw_alpha_range=powerlaw_alpha_range,
        )
    else:
        # Fall back to training energies if no heldout available
        val_gen = train_gen

    return train_gen, val_gen


def build_heldout_generator(
    mono_A: np.ndarray,
    mono_B: np.ndarray,
    energies_MeV: list[float],
    heldout: list[float],
    cfg,
    families: list[str],
    dirichlet_alpha_choices: list[float] | None = None,
) -> "SynthGenerator | None":
    """Build a SynthGenerator restricted to the held-out energies.

    Used to construct the wide-spectrum and extreme-extrapolation validation
    sets (Section 3 of the v0.4 plan). These mix ONLY held-out energies with
    generic neutral families, so they probe wide-spectrum extrapolation without
    touching the deployment spectrum. Returns None when no held-out energy is
    available.
    """
    heldout_set = set(heldout)
    held_idx = [i for i, e in enumerate(energies_MeV) if e in heldout_set]
    if not held_idx:
        return None

    if dirichlet_alpha_choices is None:
        dirichlet_alpha_choices = list(cfg.synth.get("dirichlet_alpha_choices", [0.3, 1.0, 3.0]))

    return SynthGenerator(
        mono_A[held_idx],
        mono_B[held_idx],
        np.array([energies_MeV[i] for i in held_idx]),
        families=families,
        poisson_noise=cfg.synth.poisson_noise,
        dirichlet_alpha_choices=dirichlet_alpha_choices,
        sparse_k_range=tuple(cfg.synth.get("sparse_k_range", [2, 4])),
        poisson_counts_range=tuple(cfg.synth.get("poisson_counts_range", [1000, 100000])),
        powerlaw_alpha_range=tuple(cfg.synth.get("powerlaw_alpha_range", [-3.0, 0.0])),
    )


@torch.no_grad()
def eval_emd(model, loader, criterion, device, use_amp) -> float:
    """Mean EMD of the model over a loader (in the training/target space)."""
    model.eval()
    emds = []
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            _, loss_dict = criterion(pred, y)
        emds.append(loss_dict["emd"].item())
    return float(np.mean(emds)) if emds else float("nan")


@torch.no_grad()
def eval_region_emd(
    model, loader, device, use_amp, region_mask: torch.Tensor
) -> float:
    """Mean EMD restricted to the energy region defined by region_mask.

    Both pred and target are masked to the region, then renormalized before
    computing the 1D EMD. This measures shape matching inside the region only,
    ignoring the high-energy tail.
    """
    model.eval()
    emds = []
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
        p = pred.squeeze(1).float()
        t = y.squeeze(1).float()
        mask = region_mask.to(device)
        p_r = p * mask
        t_r = t * mask
        p_r = p_r / p_r.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        t_r = t_r / t_r.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        emds.append(emd_1d(p_r, t_r).item())
    return float(np.mean(emds)) if emds else float("nan")


@torch.no_grad()
def eval_peak_error(model, loader, device, use_amp) -> float:
    """Mean |argmax(pred) - argmax(target)| in bins.

    argmax is invariant to the monotonic log-compression, so this is valid for
    both normalized (density) and non-normalized (log-compressed) targets.
    """
    model.eval()
    errs = []
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["target"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
        p = pred.squeeze(1).float().argmax(dim=-1)
        t = y.squeeze(1).float().argmax(dim=-1)
        errs.append(torch.abs(p - t).float().mean().item())
    return float(np.mean(errs)) if errs else float("nan")


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

    # --- Wide-spectrum & extreme-extrapolation validation sets (v0.4 Sec.3) ---
    # These mix ONLY held-out energies with generic neutral families. They probe
    # wide-spectrum extrapolation -- closer to the deployment scenario than the
    # mono/interpolation val set -- without ever touching the deployment file.
    wide_families = list(cfg.eval.get(
        "wide_families", ["dirichlet_uniform", "loguniform", "powerlaw_neutral"]
    ))
    extreme_families = list(cfg.eval.get(
        "extreme_families", ["dirichlet_uniform", "powerlaw_neutral"]
    ))
    extreme_alphas = list(cfg.eval.get("extreme_dirichlet_alpha_choices", [0.05, 50.0]))

    wide_gen = build_heldout_generator(
        mono_A, mono_B, energies_MeV, heldout, cfg, families=wide_families
    )
    extreme_gen = build_heldout_generator(
        mono_A, mono_B, energies_MeV, heldout, cfg,
        families=extreme_families, dirichlet_alpha_choices=extreme_alphas,
    )
    mono_held_gen = build_heldout_generator(
        mono_A, mono_B, energies_MeV, heldout, cfg, families=["mono"]
    )

    wide_val_ds = FixedMixtureSet(
        wide_gen or val_gen, input_pre, target_pre, n_samples=val_n, seed=cfg.seed + 10
    )
    extreme_val_ds = FixedMixtureSet(
        extreme_gen or val_gen, input_pre, target_pre, n_samples=val_n, seed=cfg.seed + 11
    )
    mono_held_ds = FixedMixtureSet(
        mono_held_gen or val_gen, input_pre, target_pre, n_samples=val_n, seed=cfg.seed + 12
    )

    batch_size = min(cfg.train.batch_size, samples_per_epoch)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    wide_loader = DataLoader(wide_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    extreme_loader = DataLoader(extreme_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    mono_held_loader = DataLoader(mono_held_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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

    # Build bin_dist for EMD
    emd_space = cfg.loss.get("emd_space", "index")
    bin_dist = make_bin_dist(energy_grid, emd_space)

    # Region weighting (v0.5): focus MSE on [region_kev_lo, region_kev_hi] keV.
    region_kev = list(cfg.loss.get("region_kev", []))
    region_in_weight = float(cfg.loss.get("region_in_weight", 1.0))
    region_out_weight = float(cfg.loss.get("region_out_weight", 1.0))
    rw = (
        build_region_weight(energy_grid, region_kev, region_in_weight, region_out_weight)
        if region_kev
        else None
    )
    region_mask = build_region_mask(energy_grid, region_kev) if region_kev else None

    # Loss. The mass term is only active for non-normalized heads (w_mass>0);
    # decode out of log space using the configured log_scale so the constraint
    # is on the integrated density.
    head = cfg.model.head
    w_mass = float(cfg.loss.get("w_mass", 0.0))
    target_is_log = head not in ("softmax", "softplus_renorm")
    mass_log_scale = float(cfg.preprocessing.log_scale) if target_is_log else None
    w_peak = float(cfg.loss.get("w_peak", 0.0))
    peak_window_half = int(cfg.loss.get("peak_window_half", 30))
    criterion = SpectrumLoss(
        w_mse=cfg.loss.w_mse, w_emd=cfg.loss.w_emd, bin_dist=bin_dist,
        w_mass=w_mass, log_scale=mass_log_scale,
        w_peak=w_peak, peak_window_half=peak_window_half,
        region_weight=rw,
    )

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
    best_val_emd = float("inf")
    best_select = float("inf")

    early_stop_metric = cfg.train.get("early_stop_metric", "val_emd")
    early_stop_patience = cfg.train.get("early_stop_patience", 50)
    # composite_wide = w_emd * wide_val_emd + w_peak * heldout_mono_peak_error.
    # Constrains overall wide-spectrum shape while guarding the peak position
    # (which EMD alone let regress in v0.2). No deployment-spectrum info is used.
    composite_w_emd = float(cfg.train.get("composite_w_emd", 1.0))
    composite_w_peak = float(cfg.train.get("composite_w_peak", 0.1))
    # selection_score (v0.5): peak_err + lambda_region * region_emd.
    # Peak position is the primary criterion; region EMD is a secondary shape check.
    lambda_region = float(cfg.train.get("lambda_region", 0.3))
    patience_counter = 0

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
        val_emds = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    loss, loss_dict = criterion(pred, y)
                val_losses.append(loss.item())
                val_emds.append(loss_dict["emd"].item())

        avg_val = np.mean(val_losses)
        avg_val_emd = np.mean(val_emds)
        lr = scheduler.get_last_lr()[0]

        # Wide-spectrum / extreme / peak metrics (heldout-only selection signals)
        wide_emd = eval_emd(model, wide_loader, criterion, device, use_amp)
        extreme_emd = eval_emd(model, extreme_loader, criterion, device, use_amp)
        peak_err = eval_peak_error(model, mono_held_loader, device, use_amp)
        composite = composite_w_emd * wide_emd + composite_w_peak * peak_err

        # selection_score (v0.5): peak-primary, region-EMD secondary. All computed
        # on held-out energies only; no deployment-spectrum information is used.
        if region_mask is not None:
            region_emd = eval_region_emd(
                model, wide_loader, device, use_amp, region_mask
            )
        else:
            region_emd = wide_emd
        score = peak_err + lambda_region * region_emd

        # Selection metric dispatch
        if early_stop_metric == "selection_score":
            select = score
        elif early_stop_metric == "composite_wide":
            select = composite
        else:
            select = avg_val_emd

        writer.add_scalar("Loss/train", avg_train, epoch)
        writer.add_scalar("Loss/val", avg_val, epoch)
        writer.add_scalar("EMD/val", avg_val_emd, epoch)
        writer.add_scalar("EMD/wide", wide_emd, epoch)
        writer.add_scalar("EMD/extreme", extreme_emd, epoch)
        writer.add_scalar("EMD/region", region_emd, epoch)
        writer.add_scalar("PeakErr/heldout_mono", peak_err, epoch)
        writer.add_scalar("Composite/wide", composite, epoch)
        writer.add_scalar("Score/selection", score, epoch)
        writer.add_scalar("LR", lr, epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"train={avg_train:.4f} val={avg_val:.4f} val_emd={avg_val_emd:.4f} | "
                f"wide={wide_emd:.4f} region={region_emd:.4f} peak={peak_err:.2f} "
                f"score={score:.4f} lr={lr:.2e}"
            )

        # Checkpoint
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": avg_val,
            "val_emd": avg_val_emd,
            "wide_emd": wide_emd,
            "extreme_emd": extreme_emd,
            "region_emd": region_emd,
            "heldout_peak_error": peak_err,
            "composite_wide": composite,
            "selection_score": score,
            "cfg": OmegaConf.to_container(cfg),
        }
        torch.save(state, str(ckpt_dir / "last.pt"))

        if avg_val_emd < best_val_emd:
            best_val_emd = avg_val_emd
        if avg_val < best_val_loss:
            best_val_loss = avg_val

        # Best checkpoint + early stopping on the selected metric
        if select < best_select:
            best_select = select
            patience_counter = 0
            torch.save(state, str(ckpt_dir / "best.pt"))
        else:
            patience_counter += 1
            if not fast_dev_run and patience_counter >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch+1}: "
                    f"{early_stop_metric}={select:.4f} (patience={early_stop_patience})"
                )
                break

    writer.close()
    print(f"Training done. Best {early_stop_metric}: {best_select:.4f} "
          f"(val_loss={best_val_loss:.4f}, val_emd={best_val_emd:.4f})")
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
