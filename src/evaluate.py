"""Evaluation utilities for the trained model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_spectrum_file, resample_to_canonical
from src.losses import emd_1d
from src.preprocessing import Preprocessor


def evaluate_on_set(
    model: torch.nn.Module,
    dataset,
    preprocessor: Preprocessor,
    energy_grid: np.ndarray,
    cfg,
) -> dict[str, float]:
    """Evaluate model on a dataset split.

    Parameters
    ----------
    model        : trained UNet1D
    dataset      : FixedMixtureSet or SyntheticMixtureDataset
    preprocessor : Preprocessor used during training
    energy_grid  : (N,) keV energy axis
    cfg          : OmegaConf config

    Returns
    -------
    dict with keys: emd, peak_pos_error, tail_fidelity, mse, mae
    """
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).squeeze(1)   # (N, L)
    targets = torch.cat(all_targets, dim=0).squeeze(1)

    # EMD
    emd_val = emd_1d(preds, targets).item()

    # MSE
    mse_val = torch.mean((preds - targets) ** 2).item()

    # MAE
    mae_val = torch.mean(torch.abs(preds - targets)).item()

    # Peak position error (index of max)
    pred_peak = preds.argmax(dim=-1).float()
    target_peak = targets.argmax(dim=-1).float()
    peak_err = torch.mean(torch.abs(pred_peak - target_peak)).item()

    # Tail fidelity: correlation in the tail region
    tail_pct = cfg.eval.tail_percentile
    preds_np = preds.numpy()
    targets_np = targets.numpy()

    # Compute tail threshold per sample (percentile of target)
    tail_corrs = []
    for p, t in zip(preds_np, targets_np):
        threshold = np.percentile(t, tail_pct)
        mask = t >= threshold
        if mask.sum() > 1:
            corr = np.corrcoef(p[mask], t[mask])[0, 1]
            if not np.isnan(corr):
                tail_corrs.append(corr)

    tail_fidelity = float(np.mean(tail_corrs)) if tail_corrs else float("nan")

    return {
        "emd": emd_val,
        "mse": mse_val,
        "mae": mae_val,
        "peak_pos_error": peak_err,
        "tail_fidelity": tail_fidelity,
    }


def evaluate_gcr(
    model: torch.nn.Module,
    cfg,
    preprocessor: Preprocessor,
    energy_grid: np.ndarray,
) -> dict[str, float]:
    """Evaluate model on the GCR spectrum.

    Loads GCR file, resamples to canonical grid, runs model on SiC input,
    compares predicted TEPC to true TEPC.

    Parameters
    ----------
    model        : trained UNet1D
    cfg          : OmegaConf config
    preprocessor : Preprocessor
    energy_grid  : (N,) canonical keV grid

    Returns
    -------
    dict with metrics and arrays for plotting
    """
    gcr_path = str(cfg.data.gcr_file)
    # GCR file has no skiprows needed (row 0 is header)
    spec = load_spectrum_file(gcr_path, skiprows=0)

    gcr_energy = spec["energy"]
    gcr_sic = spec["SiC_SBD"]
    gcr_tepc = spec["TEPC"]

    # Resample to canonical grid
    sic_resampled = resample_to_canonical(gcr_energy, gcr_sic, energy_grid)
    tepc_resampled = resample_to_canonical(gcr_energy, gcr_tepc, energy_grid)

    # Preprocess input
    sic_t, total = preprocessor.transform(sic_resampled)
    x = torch.from_numpy(sic_t).unsqueeze(0).unsqueeze(0)  # (1, 1, N)

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred_t = model(x.to(device)).cpu().squeeze().numpy()  # (N,)

    # Inverse transform for physical comparison
    pred_phys = preprocessor.inverse_transform(pred_t, total_counts=tepc_resampled.sum())
    # Normalize both for fair comparison
    tepc_norm = tepc_resampled / (tepc_resampled.sum() + 1e-30)
    pred_norm = pred_phys / (pred_phys.sum() + 1e-30)

    pred_tensor = torch.from_numpy(pred_norm.astype(np.float32)).unsqueeze(0)
    target_tensor = torch.from_numpy(tepc_norm.astype(np.float32)).unsqueeze(0)

    emd_val = emd_1d(pred_tensor, target_tensor).item()
    mse_val = float(np.mean((pred_norm - tepc_norm) ** 2))
    mae_val = float(np.mean(np.abs(pred_norm - tepc_norm)))

    pred_peak = int(np.argmax(pred_norm))
    true_peak = int(np.argmax(tepc_norm))
    peak_err = abs(pred_peak - true_peak)

    return {
        "emd": emd_val,
        "mse": mse_val,
        "mae": mae_val,
        "peak_pos_error": float(peak_err),
        # Arrays for plotting
        "_energy_grid": energy_grid,
        "_sic": sic_resampled,
        "_tepc_true": tepc_norm,
        "_tepc_pred": pred_norm,
    }
