"""Visualization utilities for spectrum comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum_comparison(
    energy: np.ndarray,
    A: np.ndarray,
    B_true: np.ndarray,
    B_pred: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """Plot input and target spectra with prediction overlay.

    Uses log-log axes. Saves to out_path.

    Parameters
    ----------
    energy  : (N,) energy axis (keV)
    A       : (N,) input spectrum (SiC)
    B_true  : (N,) true output spectrum (TEPC)
    B_pred  : (N,) predicted output spectrum
    title   : figure title
    out_path: save path
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Filter zeros for log-log
    eps = 1e-30
    e = np.clip(energy, eps, None)
    a = np.clip(A, eps, None)
    bt = np.clip(B_true, eps, None)
    bp = np.clip(B_pred, eps, None)

    ax.loglog(e, a, color="steelblue", alpha=0.6, label="Input (SiC)", linewidth=1.2)
    ax.loglog(e, bt, color="forestgreen", linewidth=1.8, label="True TEPC")
    ax.loglog(e, bp, color="tomato", linewidth=1.8, linestyle="--", label="Predicted TEPC")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Spectral density (a.u.)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cdf_comparison(
    energy: np.ndarray,
    B_true: np.ndarray,
    B_pred: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """Plot CDFs of true and predicted spectra.

    Parameters
    ----------
    energy  : (N,) energy axis (keV)
    B_true  : (N,) true output spectrum (TEPC)
    B_pred  : (N,) predicted output spectrum
    title   : figure title
    out_path: save path
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    cdf_true = np.cumsum(B_true)
    cdf_pred = np.cumsum(B_pred)

    # Normalize
    if cdf_true[-1] > 0:
        cdf_true = cdf_true / cdf_true[-1]
    if cdf_pred[-1] > 0:
        cdf_pred = cdf_pred / cdf_pred[-1]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogx(energy, cdf_true, color="forestgreen", linewidth=1.8, label="True TEPC CDF")
    ax.semilogx(
        energy, cdf_pred, color="tomato", linewidth=1.8, linestyle="--", label="Predicted CDF"
    )

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
