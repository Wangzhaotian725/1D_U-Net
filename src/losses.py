"""Loss functions for spectrum regression."""
from __future__ import annotations

import torch
import torch.nn as nn


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
    """Combined MSE + EMD loss for spectrum prediction.

    Parameters
    ----------
    w_mse    : weight for MSE term
    w_emd    : weight for EMD term
    bin_dist : (N,) or None; passed to emd_1d
    """

    def __init__(
        self,
        w_mse: float = 1.0,
        w_emd: float = 1.0,
        bin_dist: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.w_mse = w_mse
        self.w_emd = w_emd
        if bin_dist is not None:
            self.register_buffer("bin_dist", bin_dist)
        else:
            self.bin_dist = None

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
        (total_loss, {'mse': ..., 'emd': ...})
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

        return total, {"mse": mse.detach(), "emd": emd.detach()}
