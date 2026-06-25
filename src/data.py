"""Data loading utilities for detector spectrum files."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY_ALIASES = {"kev", "e(kev)"}
SIC_ALIASES = {"sic_sbd", "sic"}
TEPC_ALIASES = {"tepc"}


def _normalize_col(name: str) -> str:
    """Lowercase, strip spaces/underscores for alias matching."""
    return re.sub(r"[\s_]+", "_", str(name).strip().lower()).strip("_")


def _find_column(df: pd.DataFrame, aliases: set[str]) -> str | None:
    """Return the first column name that matches any alias after normalization."""
    for col in df.columns:
        if _normalize_col(col) in aliases:
            return col
    return None


def load_spectrum_file(path: str, skiprows: int = 1) -> dict:
    """Load a single spectrum Excel file.

    Returns
    -------
    dict with keys:
        'energy'      : np.ndarray (N,)  keV
        'SiC_SBD'     : np.ndarray (N,)
        'TEPC'        : np.ndarray (N,)
        'energy_MeV'  : float | None
    """
    path = str(path)
    # Parse energy from filename
    m = re.search(r"(\d+)MeV", Path(path).name, re.IGNORECASE)
    energy_MeV = float(m.group(1)) if m else None

    df = pd.read_excel(path, skiprows=skiprows)

    energy_col = _find_column(df, ENERGY_ALIASES)
    sic_col = _find_column(df, SIC_ALIASES)
    tepc_col = _find_column(df, TEPC_ALIASES)

    if energy_col is None:
        raise ValueError(f"Could not find energy column in {path}. Columns: {df.columns.tolist()}")
    if sic_col is None:
        raise ValueError(f"Could not find SiC column in {path}. Columns: {df.columns.tolist()}")
    if tepc_col is None:
        raise ValueError(f"Could not find TEPC column in {path}. Columns: {df.columns.tolist()}")

    energy = df[energy_col].values.astype(np.float64)
    sic = df[sic_col].values.astype(np.float64)
    tepc = df[tepc_col].values.astype(np.float64)

    # Try to load canonical grid for assertion (training files only)
    canonical_path = Path(__file__).parent.parent / "data" / "processed" / "energy_grid.npy"
    is_gcr = energy_MeV is None or "GCR" in Path(path).name.upper()

    if not is_gcr and canonical_path.exists():
        canonical_grid = np.load(str(canonical_path))
        if not np.allclose(energy, canonical_grid, rtol=1e-4):
            warnings.warn(
                f"{path}: energy grid does not match canonical grid.", stacklevel=2
            )

    return {
        "energy": energy,
        "SiC_SBD": sic,
        "TEPC": tepc,
        "energy_MeV": energy_MeV,
    }


def resample_to_canonical(
    energy: np.ndarray,
    values: np.ndarray,
    canonical_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate spectrum onto canonical_grid in log-energy space.

    Parameters
    ----------
    energy        : (N,) source energy axis (keV)
    values        : (N,) spectrum values (non-negative)
    canonical_grid: (M,) target energy axis (keV)

    Returns
    -------
    (M,) resampled and renormalized spectrum
    """
    log_src = np.log(np.clip(energy, 1e-30, None))
    log_dst = np.log(np.clip(canonical_grid, 1e-30, None))

    resampled = np.interp(log_dst, log_src, values, left=0.0, right=0.0)

    original_sum = values.sum()
    resampled_sum = resampled.sum()

    if original_sum > 0:
        discarded_mass = abs(original_sum - resampled_sum) / original_sum
        if discarded_mass > 1e-4:
            warnings.warn(
                f"resample_to_canonical: discarded mass fraction {discarded_mass:.4f} > 1e-4",
                stacklevel=2,
            )
        # Renormalize to preserve total mass
        if resampled_sum > 0:
            resampled = resampled * (original_sum / resampled_sum)

    return resampled
