"""Loss functions for spectrum regression."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_bin_dist(energy_grid: np.ndarray, emd_space: str) -> "torch.Tensor | None":
    """Build per-bin distance weights for EMD.

    Parameters
    ----------
    energy_grid : (N,) energy axis in keV
    emd_space   : 'index' | 'log_energy' | 'linear_energy'

    Returns
    -------
    torch.Tensor of shape (N,) or None for index space
    """
    if emd_space == "index":
        return None
    elif emd_space == "log_energy":
        log_e = np.log10(energy_grid.astype(np.float64))
        # delta between adjacent bins; pad last with same as second-to-last
        deltas = np.diff(log_e)
        deltas = np.append(deltas, deltas[-1])
        return torch.tensor(deltas, dtype=torch.float32)
    elif emd_space == "linear_energy":
        deltas = np.diff(energy_grid.astype(np.float64))
        deltas = np.append(deltas, deltas[-1])
        return torch.tensor(deltas, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown emd_space: {emd_space!r}")


def build_region_weight(
    energy_grid: np.ndarray,
    region_kev: list[float],
    in_weight: float = 1.0,
    out_weight: float = 0.1,
) -> torch.Tensor:
    """Per-bin MSE weights: high inside region_kev, low outside.

    Parameters
    ----------
    energy_grid : (N,) energy axis in keV
    region_kev  : [lo, hi] keV bounds (inclusive)
    in_weight   : weight for bins inside the region
    out_weight  : weight for bins outside the region
    """
    lo, hi = region_kev
    mask = (energy_grid >= lo) & (energy_grid <= hi)
    w = np.where(mask, in_weight, out_weight).astype(np.float32)
    return torch.tensor(w)


def build_region_mask(energy_grid: np.ndarray, region_kev: list[float]) -> torch.Tensor:
    """Boolean-style float mask (1.0 inside, 0.0 outside) for region EMD."""
    lo, hi = region_kev
    mask = (energy_grid >= lo) & (energy_grid <= hi)
    return torch.tensor(mask.astype(np.float32))


def soft_peak_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_half: int = 30,
) -> torch.Tensor:
    """Differentiable peak-alignment loss via local soft centroid.

    Finds the hard argmax of *target* (no gradient), then computes the
    soft centroid of both pred and target inside a ±window_half bin window
    around that peak. Penalizes the squared centroid offset.

    Using the target peak window (rather than a whole-spectrum soft-argmax)
    prevents the high-energy flat tail from pulling the centroid estimate away
    from the actual spectral peak.

    Parameters
    ----------
    pred, target  : (B, N) non-negative spectra (normalized or unnormalized)
    window_half   : half-width of the peak window in bins

    Returns
    -------
    scalar mean squared centroid offset
    """
    N = pred.shape[-1]
    dev = pred.device
    idx = torch.arange(N, dtype=pred.dtype, device=dev)  # (N,)

    # Hard argmax of target — used only to centre the window, no gradient needed.
    peak_idx = target.argmax(dim=-1)  # (B,)

    i = idx.unsqueeze(0)           # (1, N)
    p = peak_idx.unsqueeze(1)      # (B, 1)
    mask = ((i >= (p - window_half)) & (i <= (p + window_half))).float()  # (B, N)

    # Masked soft centroid for pred
    pred_m = pred * mask
    pred_sum = pred_m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    pred_centroid = (pred_m * idx).sum(dim=-1) / pred_sum.squeeze(-1)   # (B,)

    # Masked soft centroid for target
    tgt_m = target * mask
    tgt_sum = tgt_m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    tgt_centroid = (tgt_m * idx).sum(dim=-1) / tgt_sum.squeeze(-1)     # (B,)

    return torch.mean((pred_centroid - tgt_centroid) ** 2)


def emd_1d(
    pred: torch.Tensor,
    target: torch.Tensor,
    bin_dist: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D Earth Mover's Distance (Wasserstein-1).

    EMD = sum_i |CDF_pred(i) - CDF_target(i)| * d_i

    Parameters
    ----------
    pred     : (B, N) normalized non-negative predictions
    target   : (B, N) normalized non-negative targets
    bin_dist : (N,) per-bin weights (defaults to uniform 1/N)

    Returns
    -------
    scalar: mean EMD over batch
    """
    cdf_p = torch.cumsum(pred, dim=-1)
    cdf_t = torch.cumsum(target, dim=-1)
    diff = torch.abs(cdf_p - cdf_t)

    if bin_dist is not None:
        return (diff * bin_dist).sum(dim=-1).mean()
    return diff.sum(dim=-1).mean()


class SpectrumLoss(nn.Module):
    """Combined MSE + EMD (+ optional peak + optional mass) loss.

    Parameters
    ----------
    w_mse         : weight for MSE term
    w_emd         : weight for EMD term
    bin_dist      : (N,) or None; passed to emd_1d
    w_mass        : mass-conservation weight (for non-normalized heads only)
    log_scale     : if set, mass term decoded via expm1(x)/log_scale
    w_peak        : weight for soft peak-alignment loss (v0.5). 0 disables.
    peak_window_half : half-width (bins) of the peak centroid window
    region_weight : (N,) per-bin weights for MSE. If None, uniform weighting.
                    Use build_region_weight() to construct from an energy grid.
    w_unimodal    : weight for differentiable unimodal regulariser (v0.6). 0 disables.
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_emd: float = 1.0,
        bin_dist: torch.Tensor | None = None,
        w_mass: float = 0.0,
        log_scale: float | None = None,
        w_peak: float = 0.0,
        peak_window_half: int = 30,
        region_weight: torch.Tensor | None = None,
        w_unimodal: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_emd = w_emd
        self.w_mass = w_mass
        self.log_scale = log_scale
        self.w_peak = w_peak
        self.peak_window_half = peak_window_half
        self.w_unimodal = w_unimodal
        if bin_dist is not None:
            self.register_buffer("bin_dist", bin_dist)
        else:
            self.bin_dist = None
        if region_weight is not None:
            self.register_buffer("region_weight", region_weight)
        else:
            self.region_weight = None

    def _mass(self, x: torch.Tensor) -> torch.Tensor:
        """Integrated mass per sample. Decode out of log space if log_scale set."""
        if self.log_scale is not None:
            x = torch.expm1(x.clamp(max=20.0)) / self.log_scale
        return x.sum(dim=-1)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        Parameters
        ----------
        pred   : (B, 1, N) or (B, N)
        target : (B, 1, N) or (B, N)

        Returns
        -------
        (total_loss, {'mse': ..., 'emd': ..., 'mass': ..., 'peak': ...})
        """
        # Flatten channel dim if present
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        # MSE: apply region_weight if provided
        sq_err = (pred - target) ** 2
        if self.region_weight is not None:
            rw = self.region_weight.to(pred.device)
            mse = (sq_err * rw).mean()
        else:
            mse = sq_err.mean()

        bd = self.bin_dist if hasattr(self, "bin_dist") else None
        emd = emd_1d(pred, target, bin_dist=bd)

        total = self.w_mse * mse + self.w_emd * emd

        if self.w_mass > 0:
            mass = torch.mean((self._mass(pred) - self._mass(target)) ** 2)
            total = total + self.w_mass * mass
        else:
            mass = torch.zeros((), device=pred.device)

        if self.w_peak > 0:
            peak = soft_peak_loss(pred, target, window_half=self.peak_window_half)
            total = total + self.w_peak * peak
        else:
            peak = torch.zeros((), device=pred.device)

        if self.w_unimodal > 0:
            from src.metrics import unimodal_loss
            uni = unimodal_loss(pred)
            total = total + self.w_unimodal * uni
        else:
            uni = torch.zeros((), device=pred.device)

        return total, {
            "mse": mse.detach(),
            "emd": emd.detach(),
            "mass": mass.detach(),
            "peak": peak.detach(),
            "unimodal": uni.detach(),
        }
