"""Spectral shape diagnostics: peak counting, secondary-peak ratio, transport-leak score.

All functions operate on numpy arrays (for evaluation) or torch tensors (for differentiable
training losses). numpy-based functions are used in evaluation; torch-based in loss computation.

These metrics only depend on held-out synthetic spectra and the input SiC spectrum —
they never reference the deployment spectrum, so they are anti-leakage compliant.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Numpy-based diagnostics (for evaluation / logging)
# ---------------------------------------------------------------------------

def count_peaks(
    spectrum: np.ndarray,
    height_frac: float = 0.1,
    min_distance_bins: int = 5,
) -> int:
    """Count local maxima above height_frac * max(spectrum).

    Parameters
    ----------
    spectrum         : (N,) 1D array (non-negative)
    height_frac      : minimum peak height as fraction of global maximum
    min_distance_bins: minimum bin distance between peaks

    Returns
    -------
    Number of peaks found (0 for a flat/zero spectrum, ≥1 otherwise)
    """
    from scipy.signal import find_peaks

    if spectrum.max() == 0:
        return 0
    threshold = height_frac * spectrum.max()
    peaks, _ = find_peaks(spectrum, height=threshold, distance=min_distance_bins)
    return int(len(peaks))


def secondary_peak_ratio(
    spectrum: np.ndarray,
    height_frac: float = 0.1,
    min_distance_bins: int = 5,
) -> float:
    """Ratio of second-tallest peak height to tallest peak height.

    Returns
    -------
    0.0   → perfectly unimodal (only one peak above threshold)
    > 0.0 → secondary peak present; closer to 1.0 = more prominent second peak
    """
    from scipy.signal import find_peaks

    if spectrum.max() == 0:
        return 0.0
    threshold = height_frac * spectrum.max()
    peaks, props = find_peaks(spectrum, height=threshold, distance=min_distance_bins)
    if len(peaks) < 2:
        return 0.0
    heights = np.sort(props["peak_heights"])[::-1]
    return float(heights[1] / heights[0])


def transport_leak_score(
    pred: np.ndarray,
    inp: np.ndarray,
    height_frac: float = 0.1,
    min_distance_bins: int = 5,
) -> float:
    """Proximity of the secondary prediction peak to the input peak.

    High score → secondary peak is close to the input peak → likely skip-connection
    leakage (the input peak bleeds through to the output).

    Returns
    -------
    0.0  if pred has < 2 peaks
    value in [0, 1]: 1 = secondary pred peak exactly at input peak position
    """
    from scipy.signal import find_peaks

    # Input peak position
    if inp.max() == 0:
        return 0.0
    inp_peaks, _ = find_peaks(inp, height=height_frac * inp.max(), distance=min_distance_bins)
    inp_peak = int(inp_peaks[np.argmax(inp[inp_peaks])]) if len(inp_peaks) > 0 else int(np.argmax(inp))

    # Pred peaks
    if pred.max() == 0:
        return 0.0
    pred_peaks, props = find_peaks(pred, height=height_frac * pred.max(), distance=min_distance_bins)
    if len(pred_peaks) < 2:
        return 0.0

    # Second-tallest peak
    heights = props["peak_heights"]
    sorted_idx = np.argsort(heights)[::-1]
    second_peak = pred_peaks[sorted_idx[1]]

    N = len(pred)
    distance = abs(int(second_peak) - inp_peak) / N
    return float(1.0 - min(distance, 1.0))


@torch.no_grad()
def batch_secondary_peak_ratio(
    preds: torch.Tensor,
    height_frac: float = 0.1,
    min_distance_bins: int = 5,
) -> float:
    """Mean secondary_peak_ratio over a batch of predictions.

    Parameters
    ----------
    preds : (B, 1, N) or (B, N) tensor
    """
    if preds.dim() == 3:
        preds = preds.squeeze(1)
    arr = preds.cpu().float().numpy()
    ratios = [secondary_peak_ratio(arr[i], height_frac, min_distance_bins) for i in range(len(arr))]
    return float(np.mean(ratios))


# ---------------------------------------------------------------------------
# Torch-based differentiable unimodal regulariser (for training loss)
# ---------------------------------------------------------------------------

def unimodal_loss(pred: torch.Tensor) -> torch.Tensor:
    """Differentiable penalty for non-unimodal outputs.

    A unimodal spectrum has exactly one sign change (positive→negative) in its
    first difference. This loss penalises excess sign reversals by accumulating
    ReLU(-d[i] * d[i+1]) over all consecutive difference pairs, which is non-zero
    only where the gradient reverses direction (i.e., a local extremum).

    Parameters
    ----------
    pred : (B, N) or (B, 1, N)

    Returns
    -------
    scalar mean penalty (0 = perfectly unimodal)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    d = pred[:, 1:] - pred[:, :-1]          # (B, N-1) first differences
    sign_changes = F.relu(-d[:, :-1] * d[:, 1:])  # (B, N-2) non-zero at reversals
    return sign_changes.sum(dim=-1).mean()
