"""Tests for src/data.py"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from src.data import load_spectrum_file, resample_to_canonical

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
TRAINING_FILES = sorted(RAW_DIR.glob("*MeV.xlsx"))
GCR_FILE = Path(__file__).parent.parent / "data" / "deploy" / "GCR_spectrum.xlsx"


@pytest.mark.skipif(not TRAINING_FILES, reason="No training files found")
def test_load_training_file_shapes():
    """Training file returns shape (360,) arrays with correct keys."""
    path = str(TRAINING_FILES[0])
    spec = load_spectrum_file(path, skiprows=1)

    assert "energy" in spec
    assert "SiC_SBD" in spec
    assert "TEPC" in spec
    assert "energy_MeV" in spec

    assert spec["energy"].shape == (360,), f"Expected (360,), got {spec['energy'].shape}"
    assert spec["SiC_SBD"].shape == (360,)
    assert spec["TEPC"].shape == (360,)
    assert spec["energy_MeV"] is not None
    assert spec["energy_MeV"] > 0


@pytest.mark.skipif(not TRAINING_FILES, reason="No training files found")
def test_load_training_file_energy_MeV_parsed():
    """Filename MeV value is correctly parsed."""
    for f in TRAINING_FILES:
        spec = load_spectrum_file(str(f), skiprows=1)
        assert spec["energy_MeV"] is not None, f"Failed to parse MeV from {f.name}"
        assert spec["energy_MeV"] > 0


@pytest.mark.skipif(not GCR_FILE.exists(), reason="GCR file not found")
def test_load_gcr_file():
    """GCR file loads with alias mapping (no skiprows needed for header)."""
    spec = load_spectrum_file(str(GCR_FILE), skiprows=0)

    assert "energy" in spec
    assert "SiC_SBD" in spec
    assert "TEPC" in spec
    assert spec["energy"].shape[0] > 0
    assert spec["SiC_SBD"].shape == spec["energy"].shape
    assert spec["TEPC"].shape == spec["energy"].shape
    # GCR has no MeV in filename
    assert spec["energy_MeV"] is None


def test_resample_identity():
    """resample_to_canonical is identity when source == target grid."""
    grid = np.linspace(1.0, 100.0, 50)
    values = np.random.default_rng(0).uniform(0, 1, 50)
    resampled = resample_to_canonical(grid, values, grid)
    np.testing.assert_allclose(resampled, values, rtol=1e-4, atol=1e-8)


def test_resample_preserves_mass():
    """resample_to_canonical preserves total mass within 1e-3."""
    src_grid = np.logspace(0, 4, 200)
    canonical = np.logspace(0, 4, 360)
    rng = np.random.default_rng(42)
    values = rng.uniform(0, 1, 200)

    resampled = resample_to_canonical(src_grid, values, canonical)

    original_mass = values.sum()
    resampled_mass = resampled.sum()
    rel_err = abs(original_mass - resampled_mass) / (original_mass + 1e-30)
    assert rel_err < 1e-3, f"Mass not preserved: rel_err={rel_err:.4e}"
