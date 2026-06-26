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
    """Combined MSE + EMD (+ optional mass) loss for spectrum prediction.

    Parameters
    ----------
    w_mse     : weight for MSE term
    w_emd     : weight for EMD term
    bin_dist  : (N,) or None; passed to emd_1d
    w_mass    : weight for the mass-conservation term. 0 disables it (the default,
                appropriate for normalized heads whose output always sums to 1).
                A positive value is meant for NON-normalized heads (softplus/relu)
                which do not guarantee mass conservation.
    log_scale : if set, the mass term is computed in decoded density space
                (expm1(x)/log_scale), i.e. it constrains the integrals of the
                decoded spectra to match. If None, raw transformed sums are used.
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_emd: float = 1.0,
        bin_dist: torch.Tensor | None = None,
        w_mass: float = 0.0,
        log_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_emd = w_emd
        self.w_mass = w_mass
        self.log_scale = log_scale
        if bin_dist is not None:
            self.register_buffer("bin_dist", bin_dist)
        else:
            self.bin_dist = None

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
        (total_loss, {'mse': ..., 'emd': ..., 'mass': ...})
        """
        # Flatten channel dim if present
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        mse = torch.mean((pred - target) ** 2)

        bd = self.bin_dist if hasattr(self, "bin_dist") else None
        emd = emd_1d(pred, target, bin_dist=bd)

        total = self.w_mse * mse + self.w_emd * emd

        if self.w_mass > 0:
            mass = torch.mean((self._mass(pred) - self._mass(target)) ** 2)
            total = total + self.w_mass * mass
        else:
            mass = torch.zeros((), device=pred.device)

        return total, {"mse": mse.detach(), "emd": emd.detach(), "mass": mass.detach()}
