"""Transfer Operator architecture for spectrum-to-spectrum mapping.

Learns a non-negative, column-normalised transfer matrix T such that:
    pred = T @ input_density

Physical motivation: the TEPC response is a linear, energy-dependent operator
acting on the incident fluence. Encoding this structure directly prevents the
model from generating spurious secondary peaks, because a smooth unimodal input
passed through a smooth band-limited operator cannot split into a bimodal output.

Constraints enforced by construction:
  - Non-negative (softmax column normalisation → all entries in [0,1])
  - Column-normalised (each column sums to 1 → mass conservation)
  - Low-rank: T_raw = U @ V^T  (U, V ∈ ℝ^{N×r})  → compact parameterisation
  - Band-limited (optional): T is zeroed outside a diagonal band of half-width
    `band_halfwidth`, reflecting that energy redistribution is semi-local
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferOperator(nn.Module):
    """Low-rank, band-limited, column-normalised transfer operator.

    Parameters
    ----------
    n_bins           : number of spectral bins (default 360)
    rank             : rank of the low-rank factorisation U @ V^T
    band_halfwidth   : half-width of the diagonal band constraint in bins.
                       Entries outside |i-j| > band_halfwidth are forced to -inf
                       before the column softmax (→ effectively 0).
                       Set to None to use a full 360×360 operator.
    input_conditioned: if True, U is computed from a small encoder applied to
                       the input spectrum (FiLM-like). Not yet implemented;
                       raises if True.
    head             : output activation ('softplus_renorm' recommended)
    """

    def __init__(
        self,
        n_bins: int = 360,
        rank: int = 24,
        band_halfwidth: int | None = 60,
        input_conditioned: bool = False,
        head: str = "softplus_renorm",
    ) -> None:
        super().__init__()
        if input_conditioned:
            raise NotImplementedError("input_conditioned=True is not yet implemented")

        self.n_bins = n_bins
        self.rank = rank
        self.band_halfwidth = band_halfwidth
        self.head = head

        # Low-rank factors: T_raw = U @ V^T, shape (N, N)
        self.U = nn.Parameter(torch.randn(n_bins, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n_bins, rank) * 0.01)

        # Fixed band mask (registered as buffer so it moves with .to(device))
        if band_halfwidth is not None:
            i = torch.arange(n_bins).unsqueeze(1)   # (N, 1)
            j = torch.arange(n_bins).unsqueeze(0)   # (1, N)
            # mask[i, j] = 1 if |i-j| <= halfwidth else 0
            band_mask = (torch.abs(i - j) <= band_halfwidth).float()
        else:
            band_mask = torch.ones(n_bins, n_bins)
        self.register_buffer("band_mask", band_mask)

        # Large negative value to zero out off-band entries before softmax
        self.register_buffer(
            "neg_inf_mask",
            torch.where(band_mask == 0, torch.full_like(band_mask, -1e9), torch.zeros_like(band_mask)),
        )

    def _build_T(self) -> torch.Tensor:
        """Build the column-normalised transfer matrix.

        Returns
        -------
        T : (N, N) non-negative, each column sums to 1
        """
        T_raw = self.U @ self.V.t()               # (N, N)
        T_raw = T_raw + self.neg_inf_mask          # zero off-band entries via -inf
        T = torch.softmax(T_raw, dim=0)            # column softmax → sum_rows = 1
        return T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transfer operator.

        Parameters
        ----------
        x : (B, 1, N) input spectrum (normalised density)

        Returns
        -------
        (B, 1, N) output spectrum
        """
        L = x.shape[-1]
        x_flat = x.squeeze(1)          # (B, N)

        T = self._build_T()            # (N, N)
        # pred[b, n] = Σ_m T[n, m] * x[b, m]
        pred = x_flat @ T.t()          # (B, N)

        # Safety head (column-normalised T with normalised input already sums to 1,
        # but re-apply head for numerical robustness)
        if self.head == "softplus_renorm":
            pred = F.softplus(pred)
            pred = pred / pred.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        elif self.head == "softmax":
            pred = torch.softmax(pred, dim=-1)

        pred = pred.unsqueeze(1)       # (B, 1, N)
        return pred

    def extra_repr(self) -> str:
        return (
            f"n_bins={self.n_bins}, rank={self.rank}, "
            f"band_halfwidth={self.band_halfwidth}, head={self.head!r}"
        )
