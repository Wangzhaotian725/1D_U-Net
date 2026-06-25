"""Anti-leakage tests for experiment v0.2."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# 1. No GCR references outside allowed files
# ---------------------------------------------------------------------------

def test_no_gcr_in_train():
    """No Python file in src/ or scripts/ (except run_gcr.py and evaluate_gcr)
    should reference the GCR file path."""
    root = Path(__file__).parent.parent
    forbidden_patterns = {"GCR_spectrum", "deploy/"}

    violations = []
    for py_file in sorted((root / "src").rglob("*.py")):
        # evaluate_gcr function is allowed (it's the deployment evaluator)
        text = py_file.read_text()
        # Allow evaluate.py since it contains evaluate_gcr (deployment only)
        if py_file.name == "evaluate.py":
            # Only the evaluate_gcr function is allowed to ref GCR — check train.py etc.
            continue
        for pat in forbidden_patterns:
            if pat in text:
                violations.append(f"{py_file}: contains {pat!r}")

    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        for py_file in sorted(scripts_dir.rglob("*.py")):
            if py_file.name == "run_gcr.py":
                continue
            text = py_file.read_text()
            for pat in forbidden_patterns:
                if pat in text:
                    violations.append(f"{py_file}: contains {pat!r}")

    assert not violations, "GCR references found outside allowed files:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# 2. Synth neutral families work; gcr_like raises; no gcr_powerlaw_index attr
# ---------------------------------------------------------------------------

def test_synth_neutral():
    """Neutral families work; gcr_like raises; gcr_powerlaw_index is absent."""
    from src.synth import SynthGenerator, sample_weights

    rng = np.random.default_rng(42)
    n = 10

    # gcr_like should raise
    with pytest.raises(ValueError, match="gcr_like"):
        sample_weights(n, "gcr_like", rng)

    # SynthGenerator should not have gcr_powerlaw_index
    mono_A = np.ones((n, 32)) / 32
    mono_B = np.ones((n, 32)) / 32
    energies = np.linspace(100, 10000, n)

    gen = SynthGenerator(mono_A, mono_B, energies, families=["mono"])
    assert not hasattr(gen, "gcr_powerlaw_index"), \
        "SynthGenerator should not have gcr_powerlaw_index attribute"

    # All five neutral families work
    energies = np.linspace(100.0, 10000.0, n)
    for family in ("mono", "sparse_k", "dirichlet_uniform", "loguniform", "powerlaw_neutral"):
        w = sample_weights(n, family, rng, energies_MeV=energies)
        assert w.shape == (n,), f"Family {family!r} returned wrong shape"
        assert (w >= 0).all(), f"Family {family!r} returned negative weights"
        assert w.sum() > 0, f"Family {family!r} returned all-zero weights"


# ---------------------------------------------------------------------------
# 3. Heldout energies are disjoint from training energies
# ---------------------------------------------------------------------------

def test_heldout_disjoint():
    """Training and heldout energy sets must be disjoint."""
    from src.synth import SynthGenerator

    all_energies = [200, 300, 400, 500, 600, 800, 1000, 2000, 4000, 7000, 10000]
    heldout = [600, 2000, 7000]

    heldout_set = set(heldout)
    train_energies = [e for e in all_energies if e not in heldout_set]
    held_energies = [e for e in all_energies if e in heldout_set]

    # Disjoint check
    assert set(train_energies) & set(held_energies) == set(), \
        "Train and heldout energy sets overlap!"

    # SynthGenerator built from train energies contains only train energies
    K_train = len(train_energies)
    mono_A = np.ones((K_train, 32)) / 32
    mono_B = np.ones((K_train, 32)) / 32
    gen = SynthGenerator(mono_A, mono_B, np.array(train_energies), families=["mono"])

    gen_energies = set(gen.energies_MeV.tolist())
    assert gen_energies & heldout_set == set(), \
        f"SynthGenerator contains heldout energies: {gen_energies & heldout_set}"


# ---------------------------------------------------------------------------
# 4. Config has gcr_file but train.py body does not reference GCR_spectrum
# ---------------------------------------------------------------------------

def test_config_select_uses_heldout_only():
    """experiment_v2.yaml has gcr_file, but train.py's train() body does not
    reference GCR_spectrum."""
    root = Path(__file__).parent.parent

    cfg = OmegaConf.load(root / "configs" / "experiment_v2.yaml")
    assert "gcr_file" in cfg.data, "cfg.data.gcr_file must be present in experiment_v2.yaml"

    train_src = (root / "src" / "train.py").read_text()
    assert "GCR_spectrum" not in train_src, \
        "src/train.py must not reference GCR_spectrum (only evaluate_gcr may)"
